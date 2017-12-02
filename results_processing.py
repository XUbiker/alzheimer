import math
import numpy as np
from scipy.stats import chisquare

def top_mean_with_variance(np_data, interval, mult_factor = 1):
    if interval >= np_data.shape[0]:
        return 0.0, 0.0
    n_intervals = np_data.shape[0] - interval + 1
    means = np.zeros(n_intervals)
    vars = np.zeros(n_intervals)
    for i in range(n_intervals):
        d = np.multiply(np_data[i:i+interval], mult_factor)
        means[i] = np.mean(d)
        vars[i] = np.var(d)
    idx = np.argmax(means)
    return means[idx], vars[idx]

def p_value(predictions, expectations):
    assert predictions.shape[0] == expectations.shape[0], 'invalid input shape'
    exp_unique, exp_count = np.unique(expectations.astype(int), return_counts=True)
    pred_unique, pred_count = np.unique(predictions.astype(int), return_counts=True)
    expectations_count = dict(zip(exp_unique, exp_count))
    predictions_count = dict(zip(pred_unique, pred_count))
    _predictions = [predictions_count.get(e, 0) for e in expectations_count]
    _expectations = [expectations_count[e] for e in expectations_count]
    return chisquare(_predictions, _expectations)[1]

def confidence_interval(value, number_of_subjects, confidence_level=0.95):
    assert confidence_level == 0.95, 'unsupported confidence value'
    assert (value >= 0) and (value <= 1), 'unsupported accuracy range'
    left = value - 1.96 * math.sqrt(value * (1 - value) / number_of_subjects)
    right = value + 1.96 * math.sqrt(value * (1 - value) / number_of_subjects)
    return max(left, 0.0), min(right, 1.0)

def confusion_matrix(size, predictions, expectations, comment='', log_to_ptint=None):
    """
    Construct a confusion matrix according to expectations and predictions.
    First index corresponds to predicted value, second - to the expected one.
    """
    assert predictions.shape[0] == expectations.shape[0], 'invalid input shape'
    matrix = np.zeros((size, size), dtype=np.int32)
    for ii in range(predictions.shape[0]):
        matrix[int(predictions[ii])][int(expectations[ii])] += 1
    if log_to_ptint:
        str_representation = comment + '\n' + np.array2string(matrix)
        log_to_ptint.get().info(str_representation)
    return matrix


def bin_metric(confusion_matrix, class_index, metric):
    (tp, tn, fp, fn) = (0.0, 0.0, 0.0, 0.0)
    for ii in range(confusion_matrix.shape[0]):  # predicted
        for jj in range(confusion_matrix.shape[1]):  # expected
            if (ii == class_index) and (jj == class_index):
                tp += confusion_matrix[ii][jj]
            elif (ii == class_index) and (jj != class_index):
                fp += confusion_matrix[ii][jj]
            elif (ii != class_index) and (jj == class_index):
                fn += confusion_matrix[ii][jj]
            else:
                tn += confusion_matrix[ii][jj]
    return {
        'ACC': (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn > 0 else 0.0,
        'TPR': tp / (tp + fn) if tp + fn > 0 else 0.0,
        'TNR': tn / (tn + fp) if tn + fp > 0 else 0.0,
        'BAC': 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if (tp + fn)*(tn + fp) > 0 else 0.0
    }[metric]
