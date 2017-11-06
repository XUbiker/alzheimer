import numpy as np
from scipy.stats import chisquare

def top_mean_with_variance(np_data, interval):
    if interval >= np_data.shape[0]:
        return 0.0, 0.0
    n_intervals = np_data.shape[0] - interval + 1
    means = np.zeros(n_intervals)
    vars = np.zeros(n_intervals)
    for i in range(n_intervals):
        d = np_data[i:i+interval]
        means[i] = np.mean(d)
        vars[i] = np.var(d)
    idx = np.argmax(means)
    return means[idx], vars[idx]


def p_value(observations, expectations):
    return chisquare(observations, expectations)[1]

print(p_value([90, 60], [100, 50]))

data = np.array([1, 2, 3, 4])
tm = top_mean_with_variance(data, 2)
print(tm)


a = np.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
unique, counts = np.unique(a, return_counts=True)
print(dict(zip(unique, counts)))
