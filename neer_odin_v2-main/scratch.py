import pandas as pd
import numpy as np
import time

class ReplayClock:
    def __init__(self, start_ns, speedup=1.0):
        self.wall_start = time.time()
        self.log_start_ns = start_ns
        self.speedup = speedup

    def now_ns(self):
        return self.log_start_ns + (time.time() - self.wall_start) * self.speedup * 1e9

poses_df = pd.read_csv('/home/hello-stretch/yor_data/runs/tt_table/zed_pose.csv')
start_ns = poses_df.iloc[0].recv_ns
clock = ReplayClock(start_ns, speedup=1.0)
times = poses_df['recv_ns'].values

time.sleep(1)
now = clock.now_ns()
idx = np.searchsorted(times, now)
print(f"Elapsed 1s, now_ns={now}, first_time={times[0]}, last_time={times[-1]}, idx={idx}")
