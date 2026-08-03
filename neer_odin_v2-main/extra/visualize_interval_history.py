import matplotlib.pyplot as plt
import numpy as np

interval_history = np.fromfile("interval_history.bin", dtype=np.float32)

plt.scatter(range(len(interval_history)), interval_history)
plt.savefig("interval_history.png")
plt.close()