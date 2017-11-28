import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import ex_config as cfg
import h5py
import results_processing
import pickle
import os
from logger import XLogger

logs = sorted(l if '.' not in l else os.path.split(l)[0] for l in os.listdir("./logs/"))
log_id = 0 if not logs else int(logs[-1]) + 1
experiment_name = str(log_id).zfill(4)
log = XLogger('./logs/' + experiment_name, full_format=False)

samples = ('train', 'eval', 'test_0', 'test_1', 'test_2')
sample_to_h5_series = {'train': 'train', 'eval': 'train', 'test_0': 'test_0', 'test_1': 'test_1', 'test_2': 'test2'}
samples_eval = ('eval', 'test_0', 'test_1', 'test_2')
samples_test = ('test_0', 'test_1', 'test_2')

sets_dir = cfg.h5_cache_dir + '/sets_10_2/'
cfg_str = 'AD_NC'


class Params:
    def __init__(self):
        self.main_class_idx = cfg.get_label_code('ternary', 'AD')
        self.h5_data_path = {
            'train': sets_dir + 'alz_train_e5_' + cfg_str + '.h5',
            'test_0': sets_dir + 'alz_test_0_e5_' + cfg_str + '.h5',
            'test_1': sets_dir + 'alz_test_1_e5_' + cfg_str + '.h5',
            'test_2': sets_dir + 'alz_test_2_e5_' + cfg_str + '.h5'
        }
        # self.h5_series_path = ('data/smri_L', 'data/smri_R', 'data/md_L', 'data/md_R')
        self.h5_series_path = ('data/smri_L', 'data/smri_R')
        # self.h5_series_path = ('data/smri_LR', 'data/md_LR')
        self.n_series = len(self.h5_series_path)
        self.h5_labels_path = 'labels/labels_L'
        self.batch_size = {'train': 17, 'eval': 1, 'test_0': 1, 'test_1': 1, 'test_2': 1}
        self.start_learning_rate = 0.01
        self.decay_iterations = 100
        self.decay_rate = 0.8
        self.momentum = 0.93
        self.target_size = 3
        self.num_channels = 1
        self.generations = 50
        self.eval_every = 10
        self.cv_reshuffle_every = 500
        self.print_weights_every = 500
        self.conv_kernels = (5, 4, 3, 3, 3)
        self.pool_kernels = (2, 2, 2, 2, 2)
        self.conv_features = (16, 32, 64, 128, 128)
        self.n_conv_layers = len(self.conv_kernels)
        self.fc_features = (16,)
        self.n_fc_layers = len(self.fc_features)
        self.dropout = {'train': 0.5, 'eval': 1.0, 'test_0': 1.0, 'test_1': 1.0, 'test_2': 1.0}
        self.metric_mean_intervals = (1, 5, 10, 20)
        self.plot_type = {'train': 'k--', 'eval': 'g--', 'test_0': 'r-', 'test_1': 'b-', 'test_2': 'y-'}

    def __str__(self):
        return '\n'.join("%s: %s" % item for item in sorted(vars(self).items()))


p = Params()
log.get().info(':::Params:::\n' + str(p) + '\n')

h5 = {s: h5py.File(p.h5_data_path[sample_to_h5_series[s]], 'r') for s in samples}
data = {s: tuple(h5[s][path] for path in p.h5_series_path) for s in samples}
labels = {s: h5[s][p.h5_labels_path] for s in samples}


def cross_validation_reshuffle(train_eval_data, prc=0.9):
    n = train_eval_data.shape[0]
    _idx = np.asarray(range(n))
    np.random.shuffle(_idx)
    m = int(n * prc)
    _train_idx = _idx[:m]
    _eval_idx = _idx[m:]
    return _train_idx, _eval_idx


idx = {}
idx['train'], idx['eval'] = cross_validation_reshuffle(data['train'][0])
for s in samples_test:
    idx[s] = np.arange(data[s][0].shape[0])

input_shape = {}
input_data = {}
target = {}
img_shape = data['train'][0][0].shape
for s in samples:
    input_shape[s] = [p.batch_size[s]] + list(img_shape) + [1]
    input_data[s] = tuple(tf.placeholder(tf.float32, shape=input_shape[s]) for _ in range(p.n_series))
    target[s] = tf.placeholder(tf.int32, shape=p.batch_size[s])
keep_prob = tf.placeholder(tf.float32)


def create_var(name, shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()):
    return tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=initializer)


def calc_size_after_pooling(src_3d_shape, pool_list):
    s = src_3d_shape
    for mp in pool_list:
        s = math.ceil(s[0] / mp), math.ceil(s[1] / mp), math.ceil(s[2] / mp)
    return s[0] * s[1] * s[2]


# --- Create weight variables for convolution layers ---
cv_w = list([] for _ in range(p.n_conv_layers))
for i in range(p.n_series):
    for k in range(p.n_conv_layers):
        prev_f = 1 if k == 0 else p.conv_features[k - 1]
        curr_f = p.conv_features[k]
        conv_ker = p.conv_kernels[k]
        cv_w[k].append(create_var('W' + str(k) + '_' + str(i), shape=[conv_ker, conv_ker, conv_ker, prev_f, curr_f]))

# --- Create variable for fully-connected layers ---
fc1_in_size = calc_size_after_pooling(input_shape['train'][1:4], p.pool_kernels) * p.conv_features[-1] * p.n_series
fc_w = []
for i in range(p.n_fc_layers):
    prev_fc = fc1_in_size if i == 0 else p.fc_features[i - 1]
    curr_fc = p.fc_features[i]
    fc_w.append(create_var('fW' + str(i), shape=[prev_fc, curr_fc]))
fc_w.append(create_var('fW' + str(p.n_fc_layers), shape=[p.fc_features[-1], p.target_size]))
fc_b = create_var('fb' + str(p.n_fc_layers), shape=[p.target_size])


def print_weights(session):
    msg = 'weights values:\n'
    for ii in range(p.n_series):
        for j in range(p.n_conv_layers):
            v = session.run(cv_w[j][ii])
            msg += '\t mean: %f, std: %f\n'.format(np.asscalar(np.mean(v)), np.asscalar(np.std(v)))
    for ii in range(p.n_fc_layers):
        v = session.run(fc_w[ii])
        msg += '\t mean: %f, std: %f\n'.format(np.asscalar(np.mean(v)), np.asscalar(np.std(v)))
    log.get().info(msg)


def get_batch_norm_variables(name, inputs):
    dt = tf.float32
    sh = inputs.get_shape()[1:]
    scale, beta, mean, var = None, None, None, None
    with tf.variable_scope('BN') as scope:
        try:
            scale = tf.get_variable(name + '_s', shape=sh, dtype=dt, initializer=tf.ones_initializer())
            beta = tf.get_variable(name + '_b', shape=sh, dtype=dt, initializer=tf.zeros_initializer())
            mean = tf.get_variable(name + '_m', shape=sh, dtype=dt, initializer=tf.zeros_initializer(), trainable=False)
            var = tf.get_variable(name + '_v', shape=sh, dtype=dt, initializer=tf.ones_initializer(), trainable=False)
        except ValueError:
            scope.reuse_variables()
            scale = tf.get_variable(name + '_s')
            beta = tf.get_variable(name + '_b')
            mean = tf.get_variable(name + '_m')
            var = tf.get_variable(name + '_v')
        finally:
            return scale, beta, mean, var


def batch_normalization(name, inputs, is_training, decay=0.9):
    epsilon = 1e-3
    scale, beta, mean, var = get_batch_norm_variables(name, inputs)
    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(mean, mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(var, var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, mean, var, beta, scale, epsilon)


def conv_network(data, index, pool_sizes, batch_size, is_training):
    conv_strides = [1, 1, 1, 1, 1]
    conv_padding = 'SAME'
    conv_decay = 0.9
    layer_value = data
    for layer in range(p.n_conv_layers):
        p_s = pool_sizes[layer]
        conv = tf.nn.conv3d(layer_value, cv_w[layer][index], strides=conv_strides, padding=conv_padding)
        bn = batch_normalization('bn' + str(layer) + '_' + str(index), conv, is_training=is_training, decay=conv_decay)
        relu = tf.nn.relu(bn)
        pool = tf.nn.max_pool3d(relu, ksize=[1, p_s, p_s, p_s, 1], strides=[1, p_s, p_s, p_s, 1], padding=conv_padding)
        layer_value = pool
    return layer_value


def fusion_network(input_series, batch_size, is_training):
    conv_outputs = [tf.reshape(conv_network(data, idx, p.pool_kernels, batch_size, is_training), [batch_size, -1])
                    for idx, data in enumerate(input_series)]
    fusion_flat = tf.concat(conv_outputs, 1)
    layer_value = fusion_flat
    for ii in range(p.n_fc_layers):
        fc = tf.matmul(layer_value, fc_w[ii])
        bn = batch_normalization('fc_bn' + str(ii), fc, is_training=is_training, decay=0.9)
        relu = tf.nn.relu(bn)
        fc_dropout = tf.nn.dropout(relu, keep_prob)
        layer_value = fc_dropout
    # Last Fully Connected Layer
    final_output = tf.add(tf.matmul(layer_value, fc_w[-1]), fc_b)
    return final_output


output = {s: fusion_network(input_data[s], p.batch_size[s], is_training=(s == 'train')) for s in samples}

loss = {}
logits = {}
preds = {}
for s in samples:
    loss[s] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output[s], labels=target[s]))
    logits[s] = tf.nn.softmax(output[s])
    preds[s] = tf.argmax(logits[s], axis=1)


def get_confusion_matrix(predictions, expectations, comment='', do_print=True):
    """
    Construct a confusion matrix according to expectations and predictions.
    First index corresponds to predicted value, second - to the expected one.
    """
    assert predictions.shape[0] == expectations.shape[0], 'invalid input shape'
    matrix = np.zeros((p.target_size, p.target_size), dtype=np.int32)
    for ii in range(predictions.shape[0]):
        matrix[int(predictions[ii])][int(expectations[ii])] += 1
    if do_print:
        str_representation = comment + '\n' + np.array2string(matrix)
        log.get().info(str_representation)
    return matrix


def get_accuracy(confusion_matrix):
    return np.trace(confusion_matrix) / np.sum(confusion_matrix)


def get_bin_metric(confusion_matrix, class_index, metric):
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
        'ACC': (tp + tn) / (tp + fp + fn + tn),
        'TPR': tp / (tp + fn),
        'TNR': tn / (tn + fp),
        'BAC': 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    }[metric]


def get_p_value(predictions, expectations):
    assert predictions.shape[0] == expectations.shape[0], 'invalid input shape'
    exp_unique, exp_count = np.unique(expectations.astype(int), return_counts=True)
    pred_unique, pred_count = np.unique(predictions.astype(int), return_counts=True)
    expectations_count = dict(zip(exp_unique, exp_count))
    predictions_count = dict(zip(pred_unique, pred_count))
    _predictions = [predictions_count.get(e, 0) for e in expectations_count]
    _expectations = [expectations_count[e] for e in expectations_count]
    return results_processing.p_value(_predictions, _expectations)


def estimate_top_mean_and_variation(values, set_name, metric_name):
    str = '{} (it|mean|var)\non {}:\n'.format(metric_name, set_name)
    for i in p.metric_mean_intervals:
        m = results_processing.top_mean_with_variance(np.asarray(values), i)
        str += '{}: {:.2f} - {:.3f}\n'.format(i * p.eval_every, m[0], m[1])
    return str


# ---------- define optimization process ----------
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(p.start_learning_rate, global_step, p.decay_iterations, p.decay_rate,
                                           staircase=True)
optimizer = tf.train.MomentumOptimizer(learning_rate, p.momentum, use_nesterov=True)
train_step = optimizer.minimize(loss['train'], global_step=global_step)

# ------- Initialize Variables ----------
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# ---------- variables to save the metrics during optimization ----------
cm_metrics = ['ACC', 'TPR', 'TNR', 'BAC']
metrics = cm_metrics + ['pv', 'loss']
saved_m = {m: {s: [] for s in samples} for m in metrics}


def get_h5_data(source, indices):
    return list(map(lambda idx: np.expand_dims(source[idx], 3), indices))


def get_h5_labels(source, indices):
    return list(map(lambda idx: source[idx], indices))


# ---------- main optimization loop ----------
for i in range(p.generations):
    # ---------- CV reshuffle if needed ----------
    if (i + 1) % p.cv_reshuffle_every == 0:
        log.get().info('Performing cross-validation reshuffling')
        idx['train'], idx['eval'] = cross_validation_reshuffle(data['train'][0])
    # ---------- train ----------
    r_idx = np.random.choice(idx['train'], size=p.batch_size['train'])
    train_x = tuple(np.asarray(get_h5_data(j, r_idx)) for j in data['train'])
    train_y = np.asarray(get_h5_labels(labels['train'], r_idx))
    d = {input_data['train']: train_x, target['train']: train_y, keep_prob: p.dropout['train']}
    sess.run(train_step, feed_dict=d)
    train_loss, train_preds = sess.run([loss['train'], preds['train']], feed_dict=d)
    # ---------- print weights ----------
    if (i + 1) % p.print_weights_every == 0:
        print_weights(sess)
    # ---------- evaluate ----------
    if (i + 1) % p.eval_every == 0:
        # --- calculate accuracy for train set ---
        cm = get_confusion_matrix(train_preds, train_y, 'confusion matrix for train set', do_print=False)
        for m in cm_metrics:
            saved_m[m]['train'].append(get_bin_metric(cm, p.main_class_idx, m))
        saved_m['loss']['train'].append(train_loss)
        saved_m['pv']['train'].append(get_p_value(train_preds, train_y))
        # --- calculate accuracy on eval and test set ---
        for s in samples_eval:
            t_preds = np.zeros(idx[s].size, np.float32)
            t_targets = np.zeros(idx[s].size, np.float32)
            t_loss = np.zeros(idx[s].size, np.float32)
            for k in range(0, idx[s].size, p.batch_size[s]):
                _x = tuple(np.asarray(get_h5_data(j, (idx[s][k],))) for j in data[s])
                _y = get_h5_labels(labels[s], (idx[s][k],))
                d = {input_data[s]: _x, target[s]: _y, keep_prob: p.dropout[s]}
                _loss, _preds = sess.run([loss[s], preds[s]], feed_dict=d)
                t_targets[k * p.batch_size[s]:(k + 1) * p.batch_size[s]] = _y
                t_preds[k * p.batch_size[s]:(k + 1) * p.batch_size[s]] = _preds
                t_loss[k * p.batch_size[s]:(k + 1) * p.batch_size[s]] = _loss
            cm = get_confusion_matrix(t_preds, t_targets, 'Confusion matrix for ' + s + ' set', do_print=True)
            for m in cm_metrics:
                saved_m[m][s].append(get_bin_metric(cm, p.main_class_idx, m))
            saved_m['loss'][s].append(np.mean(t_loss))
            saved_m['pv'][s].append(get_p_value(t_preds, t_targets))
        # --- record and print results ---
        log.get().info('Generation #{}.'.format(i + 1))
        log.get().info(''.ljust(8) + '\t' + '\t'.join(s.ljust(8) for s in samples))
        for m in metrics:
            str_repr = '\t\t'.join('{:2.5f}'.format(saved_m[m][s][-1]) for s in samples)
            log.get().info(m.ljust(8) + '\t' + str_repr)
        log.get().info('\n')

for s in samples:
    h5[s].close()


def draw_plot(data_dict, eval_indices=range(0, p.generations, p.eval_every), metric_name='', exp_name=experiment_name,
              y_lim=0):
    plt.clf()
    for s in samples:
        plt.plot(eval_indices, data_dict[s], p.plot_type[s], label=s)
    plt.title(metric_name + ' per generation')
    plt.xlabel('generation')
    plt.ylabel(metric_name)
    plt.legend(loc='lower right')
    text_step = int(eval_indices[-1] * 0.75 / len(samples_test))
    text_pos = int(eval_indices[-1] * 0.05)
    for s in samples_test:
        plt.text(text_pos, 0.05, estimate_top_mean_and_variation(data_dict[s], s, metric_name), fontsize=7)
        text_pos += text_step
    plt.xlim(0)
    if y_lim > 0:
        plt.ylim(0, y_lim)
    else:
        plt.ylim(0)
    # plt.show()
    plt.savefig('./plots/{}_{}.png'.format(exp_name, metric_name), dpi=300)


# ========== Draw plots ==========
for m in metrics:
    draw_plot(saved_m[m], metric_name=m, y_lim=m in cm_metrics)

# ---------- analyze saved accuracies and p-values ----------
for s in samples:
    log.get().info(estimate_top_mean_and_variation(saved_m['ACC'][s], s.title(), 'accuracy'))
    log.get().info(estimate_top_mean_and_variation(saved_m['pv'][s], s.title(), 'p-value'))


# ---------- save the experiment results ----------
class ExperimentResults:
    def __init__(self, metrics_dict):
        self.params = p
        self.metrics_dict = metrics_dict


experiment_results = ExperimentResults({(s, m): saved_m[m][s] for m in metrics for s in samples})

with open('./logs/{}.pkl'.format(experiment_name), 'wb') as f:
    pickle.dump(experiment_results, f)

log.get().info("\nDONE!")
