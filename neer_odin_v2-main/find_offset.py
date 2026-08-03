import pandas as pd
import numpy as np
import os
import sys

sys.path.append("/home/hello-stretch/YOR")
from robot.nav.odometry.swerve_odom import SwerveOdom
from robot.slam_node_ import xyzw_xyz_to_matrix

data_dir = "/home/hello-stretch/yor_data/runs/tt_table"
zed_df = pd.read_csv(os.path.join(data_dir, "zed_pose.csv"))
enc_df = pd.read_csv(os.path.join(data_dir, "encoders.csv"))

# Odom model
odom = SwerveOdom()
odom.reset(0,0,0)
o_y = []
last_t = enc_df.iloc[0].pub_s
for i in range(len(enc_df)):
    row = enc_df.iloc[i]
    dt = max(row.pub_s - last_t, 1e-4)
    last_t = row.pub_s
    odom.update(np.array([row.steer_rad_0, row.steer_rad_1, row.steer_rad_2, row.steer_rad_3]),
                np.array([row.drive_counts_0, row.drive_counts_1, row.drive_counts_2, row.drive_counts_3]), dt)
    o_y.append(odom.get_pose()[2])

# Align by recv_ns
enc_times = enc_df.recv_ns.values
o_y = np.array(o_y)

# ZED Best mapping (based on previous turns)
T = xyzw_xyz_to_matrix(np.array([zed_df.iloc[0].base_qx, zed_df.iloc[0].base_qy, zed_df.iloc[0].base_qz, zed_df.iloc[0].base_qw, zed_df.iloc[0].base_tx, zed_df.iloc[0].base_ty, zed_df.iloc[0].base_tz]))
z_t0 = np.arctan2(T[0,2], T[2,2])

# Sample at end of log
T_L = xyzw_xyz_to_matrix(np.array([zed_df.iloc[-1].base_qx, zed_df.iloc[-1].base_qy, zed_df.iloc[-1].base_qz, zed_df.iloc[-1].base_qw, zed_df.iloc[-1].base_tx, zed_df.iloc[-1].base_ty, zed_df.iloc[-1].base_tz]))
z_tL = np.arctan2(T_L[0,2], T_L[2,2])

print(f"Odom Delta: {o_y[-1] - o_y[0]:.4f} rad")
print(f"ZED (X,Z) Delta: {z_tL - z_t0:.4f} rad")
print(f"ZED (-X,Z) Delta: {np.arctan2(-T_L[0,2], T_L[2,2]) - np.arctan2(-T[0,2], T[2,2]):.4f} rad")
