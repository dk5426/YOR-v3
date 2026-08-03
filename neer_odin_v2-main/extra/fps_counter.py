import time
import numpy as np


class FPSCounter:
    def __init__(self, name: str):
        self.name = name
        self.elapsed_time_list: list[float] = []

    def __enter__(self):
        # Called each frame to update the count and possibly compute fps
        self.enter_time = time.perf_counter_ns()

    def __exit__(self, *_):
        self.elapsed_time_list.append((time.perf_counter_ns() - self.enter_time) / 1e6)
        if len(self.elapsed_time_list) > 100:
            print(f"{self.name} average time: {np.mean(self.elapsed_time_list)} ms")
            self.elapsed_time_list = []