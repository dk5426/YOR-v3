import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import threading

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from robot.slam_node_ import Slam, _quat_to_matrix, _matrix_to_quat, xyzw_xyz_to_matrix
import robot.slam_node_ 

class SharedSimState:
    current_ns = 0
    ended = False
    
class ReplayClock:
    def __init__(self, start_ns, speedup=1.0):
        self.wall_start = time.time()
        self.log_start_ns = start_ns
        self.speedup = speedup

    def wait_until(self, target_ns):
        if self.speedup <= 0: return 
        now_ns = self.log_start_ns + (time.time() - self.wall_start) * self.speedup * 1e9
        if target_ns > now_ns:
            sleep_s = (target_ns - now_ns) / 1e9 / self.speedup
            if sleep_s > 0.001:
                time.sleep(sleep_s)

class OfflineZedSub:
    def __init__(self, data_dir, clock, topics=None, host="", port=0, up_axis="y"):
        self.poses_df = pd.read_csv(os.path.join(data_dir, "zed_pose.csv"))
        self.pcd_dir = os.path.join(data_dir, "zed_pcd")
        self.clock = clock
        self._up_axis = str(up_axis).lower()
        self.idx = 0
        self.total = len(self.poses_df)
        self.is_mapping = topics is not None and "zed/pcd" in topics
        
        self._zup_to_yup = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def ready(self): return True
    def stop(self): pass
    def get_camera_info(self): return None
    def get_tracking_confidence(self): return 100.0

    def _zup_to_yup_transform(self, T): return self._zup_to_yup @ T @ self._zup_to_yup.T
    def _zup_to_yup_pose(self, qt):
        q, t = qt[:4], qt[4:7]
        T = np.eye(4, dtype=np.float32)
        T[:3,:3], T[:3,3] = _quat_to_matrix(q), t
        T = self._zup_to_yup_transform(T)
        return np.concatenate([_matrix_to_quat(T[:3,:3]), T[:3,3].astype(np.float32)])
    def _zup_to_yup_points(self, pcd):
        pcd = np.asarray(pcd)
        if pcd.ndim < 2 or pcd.shape[-1] < 3: return pcd
        xyz = np.stack([pcd[...,0], pcd[...,2], -pcd[...,1]], axis=-1)
        if pcd.shape[-1] > 3:
            out = pcd.copy()
            out[...,:3] = xyz
            return out
        return xyz

    def get_pcd_pose(self):
        if self.idx >= self.total:
            SharedSimState.ended = True
            time.sleep(1.0)
            row = self.poses_df.iloc[-1]
            return np.zeros((1,3), dtype=np.float32), np.array([row.cam_qx, row.cam_qy, row.cam_qz, row.cam_qw, row.cam_tx, row.cam_ty, row.cam_tz]), 100.0

        row = self.poses_df.iloc[self.idx]
        self.clock.wait_until(row.recv_ns)
        SharedSimState.current_ns = row.recv_ns 

        qt = np.array([row.cam_qx, row.cam_qy, row.cam_qz, row.cam_qw, row.cam_tx, row.cam_ty, row.cam_tz], dtype=np.float32)
        pcd = np.load(os.path.join(self.pcd_dir, f"{self.idx:06d}.npz"))["points"]
        
        if self._up_axis == "z":
            qt = self._zup_to_yup_pose(qt)
            pcd = self._zup_to_yup_points(pcd)
            
        self.idx += 1
        return pcd, qt, float(row.get("confidence", 100))

    def get_pose(self):
        now_ns = SharedSimState.current_ns
        idx = 0 if now_ns == 0 else min(np.searchsorted(self.poses_df['recv_ns'].values, now_ns), self.total - 1)
        row = self.poses_df.iloc[idx]

        qt = np.array([row.base_qx, row.base_qy, row.base_qz, row.base_qw, row.base_tx, row.base_ty, row.base_tz])
        T = xyzw_xyz_to_matrix(qt)
        if self._up_axis == "z":
            T = self._zup_to_yup_transform(T)
        return T[:3,3].astype(np.float32), float(np.arctan2(T[0,2], T[2,2])), T


class OfflineRPCClient:
    def __init__(self, data_dir, clock, host, port): pass
    def follow_path(self, path): pass
    def get_base_encoders(self): return None

zed_traj, enc_traj, fused_traj = [], [], []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="yor_data/runs/xxx directory")
    parser.add_argument("--speedup", type=float, default=1.0)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-ekf", dest="ekf", action="store_false")
    args = parser.parse_args()

    poses_df = pd.read_csv(os.path.join(args.data_dir, "zed_pose.csv"))
    clock = ReplayClock(poses_df.iloc[0].recv_ns, args.speedup)

    def MockZedSubFactory(**kw): return OfflineZedSub(args.data_dir, clock=clock, **kw)
    def MockRPCClientFactory(h, p): return OfflineRPCClient(args.data_dir, clock=clock, host=h, port=p)

    robot.slam_node_.ZedSub = MockZedSubFactory
    robot.slam_node_.RPCClient = MockRPCClientFactory
    
    if args.ekf:
        enc_df = pd.read_csv(os.path.join(args.data_dir, "encoders.csv"))
        
        # Repair logger glitches (CAN interleaving spikes)
        c_cols = ['drive_counts_0', 'drive_counts_1', 'drive_counts_2', 'drive_counts_3']
        c_vals = enc_df[c_cols].values
        for i in range(1, len(c_vals)):
            if np.any(np.abs(c_vals[i] - c_vals[i-1]) > 2.0):
                c_vals[i] = c_vals[i-1]
        enc_df[c_cols] = c_vals

        enc_rows = [r._asdict() for r in enc_df.itertuples()]
        enc_state = {'idx': 0}

        # Halt background threads natively
        orig_init = robot.slam_node_.EKFSlamSource.__init__
        def Patched_EKFInit(self, *a, **k):
            orig_init(self, *a, **k)
            self._stop_evt.set()
        robot.slam_node_.EKFSlamSource.__init__ = Patched_EKFInit
        
        robot.slam_node_.EKFSlamSource._orig_apply_zed_update = robot.slam_node_.EKFSlamSource._apply_zed_update
        robot.slam_node_.EKFSlamSource._apply_zed_update = lambda *a, **k: None

        # Fully synchronous trace via MapManager thread hooks
        orig_pcd = robot.slam_node_.EKFSlamSource.get_pcd_pose
        def Patched_PcdPose(self):
            # 1. Yield points & advance SharedSimState.current_ns exactly 1 step
            pcd, qt, conf = self._zed.get_pcd_pose()
            now_ns = SharedSimState.current_ns
            
            # 2. Advance encoders strictly up to this point
            idx = enc_state['idx']
            while idx < len(enc_rows) and enc_rows[idx]['recv_ns'] <= now_ns:
                row = enc_rows[idx]
                idx += 1
                dt = (row['pub_s'] - self._last_enc_t) if self._last_enc_t else 0.05
                self._last_enc_t = row['pub_s']
                dt = float(np.clip(dt, 1e-4, 0.5))
                
                steer = np.array([row['steer_rad_0'], row['steer_rad_1'], row['steer_rad_2'], row['steer_rad_3']])
                counts = np.array([row['drive_counts_0'], row['drive_counts_1'], row['drive_counts_2'], row['drive_counts_3']])
                with self._lock:
                    self._ekf.predict(self._odom.update(steer, counts, dt), dt)
            enc_state['idx'] = idx
            
            # 3. Apply ZED correctly with BASE coords
            base_trans, base_yaw, _ = self._zed.get_pose()
            self._orig_apply_zed_update(base_trans, base_yaw, confidence=conf)
            
            # 4. Save to arrays directly inline! No missed samples!
            zed_traj.append([base_trans[0], base_trans[2]])
            with self._lock:
                state, odom = self._ekf.get_state(), self._odom.get_pose()
            fused_traj.append([state[0], state[1]])
            enc_traj.append([odom[0], odom[1]])
            
            return pcd, qt, conf
        robot.slam_node_.EKFSlamSource.get_pcd_pose = Patched_PcdPose

    else:
        # Non-EKF mode, we still want to save ZED trajectories inline
        def patched_pcd_no_ekf(self):
            pcd, qt, conf = self.get_pcd_pose_orig()
            base_trans, _, _ = self.get_pose()
            zed_traj.append([base_trans[0], base_trans[2]])
            return pcd, qt, conf
        OfflineZedSub.get_pcd_pose_orig = OfflineZedSub.get_pcd_pose
        OfflineZedSub.get_pcd_pose = patched_pcd_no_ekf

    print(f"\n[OfflineSlam] Starting SLAM (Speedup: {args.speedup}x)")
    slam = Slam(target_hz=0.0, duration_s=0.0, load_map=False, save_map=args.save, map_path=None, use_ekf=args.ekf, zed_up_axis="y")
    
    t0 = time.time()
    def shutdown_monitor():
        while slam.running:
            if SharedSimState.ended:
                print("\nEnd of dataset reached! Shutting down...")
                slam.running = False
            time.sleep(0.5)
    threading.Thread(target=shutdown_monitor, daemon=True).start()

    try:
        slam.run()
    except Exception as e: print("Run Exception:", e)

    print(f"Finished processing. CPU Wall time: {time.time()-t0:.2f}s")
    if not zed_traj and hasattr(slam, "datastream"):
        print("Trajectory empty: saving manual offline snapshot if mapping failed...")
    
if __name__ == "__main__":
    main()
