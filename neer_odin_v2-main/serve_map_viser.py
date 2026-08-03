#!/usr/bin/env python3
import numpy as np
import viser
import time
import sys
import os

def main():
    map_path = "orbv2/maps/lab_m03_0529110907.npz"
    if len(sys.argv) > 1:
        map_path = sys.argv[1]

    if not os.path.exists(map_path):
        print(f"Error: Map file not found at '{map_path}'")
        sys.exit(1)

    print(f"Loading map from '{map_path}'...")
    data = np.load(map_path)
    keys = data["keys"]
    log_odds = data["log_odds"]
    color = data["color"]
    centroid = data["centroid"]
    vs = float(data["voxel_size"][0])

    # Filter out noisy voxels (log_odds >= 1.0 is stable map features)
    mask = log_odds >= 1.0
    pts = centroid[mask]
    cols = color[mask] / 255.0  # Normalize to [0, 1] for viser
    keys_filtered = keys[mask]

    print(f"Total voxels: {len(log_odds)}")
    print(f"Filtered voxels (log_odds >= 1.0): {len(pts)}")

    # Start Viser server
    port = 8099
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.scene.set_up_direction("+y")

    # Shift centroids close to origin for better rotation and camera control
    centroid_mean = np.mean(pts, axis=0)
    # We keep the raw coordinates but center the camera target around the map mean
    print(f"Centering viewer target at: {centroid_mean}")

    # Add the 3D point cloud
    server.scene.add_point_cloud(
        "map/points",
        points=pts.astype(np.float32),
        colors=cols.astype(np.float32),
        point_size=vs,
        point_shape="circle"
    )

    # Add a floor grid for perspective context
    server.scene.add_grid(
        "grid",
        width=30.0,
        height=30.0,
        plane="xz",
        position=(centroid_mean[0], np.min(pts[:, 1]), centroid_mean[2]),
        cell_color=(100, 100, 100),
        section_color=(140, 140, 140),
    )

    # Register click handler to capture clicked voxel coordinates
    click_idx = 0

    @server.on_scene_pointer(event_type="click")
    def _on_click(ev):
        nonlocal click_idx
        if ev.ray_origin is None or ev.ray_direction is None:
            return

        ray_o = np.array(ev.ray_origin, dtype=np.float32)
        ray_d = np.array(ev.ray_direction, dtype=np.float32)

        # Normalize ray direction
        d = ray_d / np.linalg.norm(ray_d)

        # Vector from ray origin to all centroids
        V = pts - ray_o
        t = np.dot(V, d)

        # Filter to points in front of the ray
        valid = t > 0
        if not np.any(valid):
            print("[Viser Click] No voxels found in front of click ray.")
            return

        V_valid = V[valid]
        t_valid = t[valid]
        pts_valid = pts[valid]
        keys_valid = keys_filtered[valid]

        # Projection vectors
        proj = t_valid[:, np.newaxis] * d

        # Distance squared to the ray
        dist_sq = np.sum((V_valid - proj) ** 2, axis=1)

        # Find closest voxel
        min_idx = np.argmin(dist_sq)
        min_dist = np.sqrt(dist_sq[min_idx])

        # Voxel size threshold to consider it a hit (e.g., within 0.15m)
        if min_dist < 0.15:
            clicked_pt = pts_valid[min_idx]
            k = keys_valid[min_idx]

            print(f"\n🎯 [Clicked Voxel (Slot {click_idx})]")
            print(f"  Voxel Index (Key): [{k[0]}, {k[1]}, {k[2]}]")
            print(f"  World Position:    [{clicked_pt[0]:.4f}, {clicked_pt[1]:.4f}, {clicked_pt[2]:.4f}]")
            print(f"  Distance to ray:   {min_dist:.4f} m")

            # Add sphere marker at clicked voxel
            server.scene.add_icosphere(
                f"map/clicked_voxel_marker_{click_idx}",
                radius=vs * 1.2,
                color=(255, 40, 80),
                position=clicked_pt
            )

            # Add text label above clicked voxel
            server.scene.add_label(
                f"map/clicked_voxel_label_{click_idx}",
                text=f"Key: [{k[0]}, {k[1]}, {k[2]}]\nWorld: [{clicked_pt[0]:.3f}, {clicked_pt[1]:.3f}, {clicked_pt[2]:.3f}]",
                position=clicked_pt + np.array([0.0, vs * 2.0, 0.0]),
                font_size_mode="scene",
                font_scene_height=0.06,
                anchor="bottom-center"
            )

            # Advance the cyclic counter for the next click (0 -> 1 -> 0 -> 1)
            click_idx = (click_idx + 1) % 2
        else:
            print(f"[Viser Click] Clicked too far from any voxel (closest dist = {min_dist:.2f} m).")

    print(f"\n[Viser] Interactive 3D visualizer running at http://localhost:{port}")
    print("[Viser] Click on any voxel to see its index and world coordinates.")
    print("[Viser] Press Ctrl+C in this terminal to stop the server.")

    # Keep server alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping Viser server.")

if __name__ == "__main__":
    main()
