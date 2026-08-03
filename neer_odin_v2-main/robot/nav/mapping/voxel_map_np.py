"""Numpy/cv2 incremental voxel map with dynamic-object removal — no ROS, no torch.

This is the same engine validated in the Odin-Nav-Stack Viser bridge, extracted
into neer_odin as the canonical non-ROS implementation. Semantics are ported
from neer_slam/robot/nav/mapping/voxel_map.py (GlobalVoxelMap):

  * log-odds occupancy: l_hit=0.85, l_min=-2.0, l_max=3.5
  * Range-Image Differencing free-space carving (_clear_dynamic_objects):
    conflict A (return behind voxel), conflict B (confirmed miss in the active
    field), occlusion guard — identical margins.
  * live->permanent promotion (promote_age) + TTL decay: probationary voxels
    not re-observed in time vanish, killing walking-person trails without
    requiring the sensor to look back.

Coordinates: world frame, right-handed Z-up / X-forward (Odin native).
The camera pose is the Odin odometry body pose (REP-103).
"""

import numpy as np
import cv2

# body (x fwd, y left, z up) -> camera (x right, y down, z fwd)
_BODY_TO_CAM = np.array([[0., -1., 0.],
                         [0., 0., -1.],
                         [1., 0., 0.]], dtype=np.float32)


def quat_to_rot(wxyz):
    """(w,x,y,z) quaternion -> 3x3 rotation matrix (body->world)."""
    w, x, y, z = wxyz
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z),     s * (x * y - z * w),     s * (x * z + y * w)],
        [    s * (x * y + z * w), 1 - s * (x * x + z * z),     s * (y * z - x * w)],
        [    s * (x * z - y * w),     s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ], dtype=np.float32)


def pack_voxel_keys(pts, voxel):
    """One int64 hash per point's voxel cell (21 bits/axis, offset non-negative)."""
    q = np.floor(pts / voxel).astype(np.int64) + (1 << 20)
    q = np.clip(q, 0, (1 << 21) - 1)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


class VoxelMap:
    """Incremental voxel map with log-odds occupancy + dynamic-object carving."""

    L_HIT, L_MIN, L_MAX = 0.85, -2.0, 3.5
    # virtual carver camera (projection model only; both sides use the same one)
    CW, CH = 320, 240
    CFX = CFY = 131.25          # == ZED default 525*0.25 -> ~101 deg HFOV
    CCX, CCY = 160.0, 120.0
    FIELD_K = 31                # px active-field dilation
    PROTECT_K = 21              # px occlusion-guard window

    def __init__(self, voxel, carve_range=4.5):
        self.voxel = voxel
        self.carve_range = float(carve_range)
        n0 = 0
        self.keys = np.empty(n0, np.int64)
        self.pts = np.empty((n0, 3), np.float32)
        self.cols = np.empty((n0, 3), np.uint8)
        self.lo = np.empty(n0, np.float32)
        self.alive = np.empty(n0, bool)
        self.age = np.empty(n0, np.int32)
        self.last_seen = np.empty(n0, np.float64)
        self._sorted = np.empty(n0, np.int64)
        self._sort_idx = np.empty(n0, np.int64)
        self._n_sorted = 0
        self.cand_keys = np.empty(0, np.int64)
        self.cand_hits = np.empty(0, np.int32)
        self.carved_total = 0
        self.decayed_total = 0

    def _lookup(self, q):
        found = np.zeros(len(q), bool)
        idx = np.zeros(len(q), np.int64)
        if self._n_sorted:
            pos = np.clip(np.searchsorted(self._sorted, q), 0, self._n_sorted - 1)
            hit = self._sorted[pos] == q
            found |= hit
            idx[hit] = self._sort_idx[pos[hit]]
        tail = self.keys[self._n_sorted:]
        if tail.size:
            spos = np.argsort(tail, kind="stable")
            st = tail[spos]
            pos = np.clip(np.searchsorted(st, q), 0, len(st) - 1)
            hit = (st[pos] == q) & ~found
            found |= hit
            idx[hit] = self._n_sorted + spos[pos[hit]]
        return found, idx

    def _append(self, keys, pts, cols, lo, age, now):
        self.keys = np.concatenate((self.keys, keys))
        self.pts = np.vstack((self.pts, pts))
        self.cols = np.vstack((self.cols, cols))
        self.lo = np.concatenate((self.lo, np.full(len(keys), lo, np.float32)))
        self.alive = np.concatenate((self.alive, np.ones(len(keys), bool)))
        self.age = np.concatenate((self.age, np.full(len(keys), age, np.int32)))
        self.last_seen = np.concatenate((self.last_seen, np.full(len(keys), now, np.float64)))
        if len(self.keys) - self._n_sorted > 100_000:
            self._sort_idx = np.argsort(self.keys, kind="stable")
            self._sorted = self.keys[self._sort_idx]
            self._n_sorted = len(self.keys)

    @property
    def n_alive(self):
        return int(self.alive.sum()) if len(self.alive) else 0

    def alive_points(self):
        return self.pts[self.alive], self.cols[self.alive]

    def update(self, pts, cols, min_hits, now):
        """Ingest one frame (already range-filtered). Returns (new_pts, new_cols)
        of voxels that just became visible (confirmed or revived), or None."""
        if not len(pts):
            return None
        uk, uidx = np.unique(pack_voxel_keys(pts, self.voxel), return_index=True)
        pts, cols = pts[uidx].astype(np.float32), cols[uidx].astype(np.uint8)

        found, fidx = self._lookup(uk)
        out_pts, out_cols = [], []

        if found.any():
            ei = fidx[found]
            was_dead = ~self.alive[ei]
            self.lo[ei] = np.minimum(self.lo[ei] + self.L_HIT, self.L_MAX)
            self.age[ei] += 1
            self.last_seen[ei] = now
            revive = was_dead & (self.lo[ei] > 0.0)
            if revive.any():
                ri = ei[revive]
                self.alive[ri] = True
                self.age[ri] = min_hits
                self.pts[ri] = pts[found][revive]
                self.cols[ri] = cols[found][revive]
                out_pts.append(self.pts[ri].copy())
                out_cols.append(self.cols[ri].copy())

        nk, npts, ncols = uk[~found], pts[~found], cols[~found]
        if len(nk):
            if min_hits <= 1:
                self._append(nk, npts, ncols, self.L_HIT, 1, now)
                out_pts.append(npts)
                out_cols.append(ncols)
            else:
                if self.cand_keys.size:
                    pos = np.clip(np.searchsorted(self.cand_keys, nk), 0, len(self.cand_keys) - 1)
                    seen = self.cand_keys[pos] == nk
                else:
                    pos = np.zeros(len(nk), int)
                    seen = np.zeros(len(nk), bool)
                promote = np.zeros(len(nk), bool)
                if seen.any():
                    ci = pos[seen]
                    self.cand_hits[ci] += 1
                    promote[seen] = self.cand_hits[ci] >= min_hits
                if promote.any():
                    self._append(nk[promote], npts[promote], ncols[promote],
                                 self.L_HIT * min_hits, min_hits, now)
                    out_pts.append(npts[promote])
                    out_cols.append(ncols[promote])
                    drop = np.zeros(len(self.cand_keys), bool)
                    drop[pos[promote]] = True
                    self.cand_keys, self.cand_hits = self.cand_keys[~drop], self.cand_hits[~drop]
                new = ~seen
                if new.any():
                    self.cand_keys = np.concatenate((self.cand_keys, nk[new]))
                    self.cand_hits = np.concatenate((self.cand_hits,
                                                     np.ones(int(new.sum()), np.int32)))
                    order = np.argsort(self.cand_keys, kind="stable")
                    self.cand_keys, self.cand_hits = self.cand_keys[order], self.cand_hits[order]
                if self.cand_keys.size > 400_000:
                    keep = self.cand_hits >= 2
                    self.cand_keys, self.cand_hits = self.cand_keys[keep], self.cand_hits[keep]

        if not out_pts:
            return None
        return np.vstack(out_pts), np.vstack(out_cols)

    def decay(self, now, promote_age=15, ttl=2.0):
        """Kill probationary voxels not re-observed within ttl seconds."""
        if not len(self.keys):
            return 0
        kill = self.alive & (self.age < promote_age) & (now - self.last_seen > ttl)
        n = int(kill.sum())
        if n:
            self.lo[kill] = self.L_MIN
            self.alive[kill] = False
            self.decayed_total += n
        return n

    def carve(self, frame_pts_world, cam_wxyz, cam_t):
        """Range-Image Differencing: clear voxels contradicted by this frame."""
        n_map = len(self.keys)
        if n_map == 0 or not len(frame_pts_world):
            return 0
        W, H = self.CW, self.CH
        fx, fy, cx, cy = self.CFX, self.CFY, self.CCX, self.CCY

        Rwb = quat_to_rot(cam_wxyz)
        Rcw = _BODY_TO_CAM @ Rwb.T
        t = np.asarray(cam_t, np.float32)

        pc = (frame_pts_world - t) @ Rcw.T
        infront = pc[:, 2] > 0.1
        pc = pc[infront]
        depth = np.full(H * W, 20.0, np.float32)
        if len(pc):
            u = (pc[:, 0] * fx / pc[:, 2] + cx).astype(np.int32)
            v = (pc[:, 1] * fy / pc[:, 2] + cy).astype(np.int32)
            ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            np.minimum.at(depth, v[ok] * W + u[ok], pc[ok, 2])
        depth = depth.reshape(H, W)
        observed = depth < 19.0

        k3 = np.ones((3, 3), np.uint8)
        depth_eroded = cv2.erode(depth, k3)
        obs_dilated = cv2.dilate(observed.astype(np.uint8), k3) > 0
        kf = np.ones((self.FIELD_K, self.FIELD_K), np.uint8)
        active_field = cv2.dilate(observed.astype(np.uint8), kf) > 0
        kp = np.ones((self.PROTECT_K, self.PROTECT_K), np.uint8)
        depth_near = cv2.erode(depth, kp)
        obs_near = cv2.dilate(observed.astype(np.uint8), kp) > 0

        ai = np.where(self.alive)[0]
        if not len(ai):
            return 0
        vc = (self.pts[ai] - t) @ Rcw.T
        infront = vc[:, 2] > 0.1
        ai, vc = ai[infront], vc[infront]
        if not len(ai):
            return 0
        u = (vc[:, 0] * fx / vc[:, 2] + cx).astype(np.int32)
        v = (vc[:, 1] * fy / vc[:, 2] + cy).astype(np.int32)
        on = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ai, u, v, z = ai[on], u[on], v[on], vc[on, 2]
        if not len(ai):
            return 0

        measured = depth_eroded[v, u]
        is_obs = obs_dilated[v, u]

        margin_a = 0.25 + 0.04 * np.clip(z, 0.0, 5.0)
        conflict_a = is_obs & (measured > 0.1) & (measured > z + margin_a)

        is_miss = measured > 19.0
        conflict_b = (~is_obs) & is_miss & active_field[v, u] & \
                     (z + 0.25 < self.carve_range)

        occ_margin = 0.15 + 0.04 * np.clip(z, 0.0, 5.0)
        occluded = obs_near[v, u] & (depth_near[v, u] > 0.1) & \
                   (z > depth_near[v, u] + occ_margin)

        clear = (conflict_a | conflict_b) & ~occluded
        ci = ai[clear]
        if len(ci):
            self.lo[ci] = self.L_MIN
            self.alive[ci] = False
            self.carved_total += len(ci)
        return int(len(ci))
