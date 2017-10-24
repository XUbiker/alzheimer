import numpy as np

def top_mean(np_data, interval):
    if interval >= np_data.shape[0]:
        return 0.0
    n_means = np_data.shape[0] - interval + 1
    means = np.zeros(n_means)
    for i in range(n_means):
        means[i] = np.mean(np_data[i:i+interval])
    return np.max(means)

data = np.array([1, 2, 3, 4])
tm = top_mean(data, 1)
print(tm)
