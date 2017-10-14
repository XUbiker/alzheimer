import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import ex_config as cfg
import h5py
from logger import XLogger

import os

logs = sorted(os.listdir("./logs/"))
log_id = 0 if not logs else int(logs[-1])+1
experiment_name = str(log_id).zfill(4)
log = XLogger('./logs/' + experiment_name, full_format=False)


class Params:
    def __init__(self):
        self.h5_train_path = cfg.h5_cache_dir + '/rois_10/alz_train_eval_AD_NC.h5'
        self.h5_test_path = cfg.h5_cache_dir + '/rois_10/alz_test_ext_AD_NC.h5'
        self.h5_series_path = ('data/smri_L', 'data/smri_R', 'data/md_L', 'data/md_R')
        self.n_series = len(self.h5_series_path)
        self.h5_labels_path = 'labels/labels_L'
        self.train_batch_size = 10
        self.eval_batch_size = 1
        self.test_batch_size = 1
        self.learning_rate = 0.005
        self.target_size = 3
        self.num_channels = 1
        self.generations = 1500
        self.eval_every = 20
        self.cv_reshuffle_every = 500
        self.print_weights_every = 500
        self.conv_kernels = (5, 4, 3, 3, 3)
        self.pool_kernels = (2, 2, 2, 2, 3)
        self.conv_features = (16, 32, 64, 128, 256)
        self.n_conv_layers = len(self.conv_kernels)
        self.fc_features = (8,)
        self.n_fc_layers = len(self.fc_features)
        self.dropout_train = 0.5
        self.dropout_eval = 1.0
        self.dropout_test = 1.0

    def __str__(self):
        return '\n'.join("%s: %s" % item for item in sorted(vars(self).items()))


p = Params()
log.get().info(':::Params:::\n' + str(p) + '\n')

train_eval_h5 = h5py.File(p.h5_train_path, 'r')
test_h5 = h5py.File(p.h5_test_path, 'r')

train_data = tuple(train_eval_h5[path] for path in p.h5_series_path)
test_data = tuple(test_h5[path] for path in p.h5_series_path)
train_labels = train_eval_h5[p.h5_labels_path]
test_labels = test_h5[p.h5_labels_path]


def cross_validation_reshuffle(train_eval_data, prc=0.9):
    n = train_eval_data.shape[0]
    idx = np.asarray(range(n))
    np.random.shuffle(idx)
    m = int(n * prc)
    _train_idx = idx[:m]
    _eval_idx = idx[m:]
    return _train_idx, _eval_idx


train_idx, eval_idx = cross_validation_reshuffle(train_data[0])

img_shape = train_data[0][0].shape
train_input_shape = [p.train_batch_size] + list(img_shape) + [1]
eval_input_shape = [p.eval_batch_size] + list(img_shape) + [1]
test_input_shape = [p.test_batch_size] + list(img_shape) + [1]

train_input = tuple(tf.placeholder(tf.float32, shape=train_input_shape) for _ in range(p.n_series))
eval_input = tuple(tf.placeholder(tf.float32, shape=eval_input_shape) for _ in range(p.n_series))
test_input = tuple(tf.placeholder(tf.float32, shape=test_input_shape) for _ in range(p.n_series))
train_target = tf.placeholder(tf.int32, shape=p.train_batch_size)
eval_target = tf.placeholder(tf.int32, shape=p.eval_batch_size)
test_target = tf.placeholder(tf.int32, shape=p.test_batch_size)
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
    for l in range(p.n_conv_layers):
        prev_f = 1 if l == 0 else p.conv_features[l - 1]
        curr_f = p.conv_features[l]
        conv_ker = p.conv_kernels[l]
        cv_w[l].append(create_var('W' + str(l) + '_' + str(i), shape=[conv_ker, conv_ker, conv_ker, prev_f, curr_f]))

# --- Create variable for fully-connected layers ---
fc1_in_size = calc_size_after_pooling(train_input_shape[1:4], p.pool_kernels) * p.conv_features[-1] * p.n_series
fc_w = []
for i in range(p.n_fc_layers):
    prev_fc = fc1_in_size if i == 0 else p.fc_features[i-1]
    curr_fc = p.fc_features[i]
    fc_w.append(create_var('fW' + str(i), shape=[prev_fc, curr_fc]))
fc_w.append(create_var('fW' + str(p.n_fc_layers), shape=[p.fc_features[-1], p.target_size]))
fc_b = create_var('fb' + str(p.n_fc_layers), shape=[p.target_size])


def print_weights(session):
    msg = 'weights values:\n'
    for i in range(p.n_series):
        for j in range(p.n_conv_layers):
            v = session.run(cv_w[j][i])
            msg += '\t mean: %f, std: %f\n'.format(np.asscalar(np.mean(v)), np.asscalar(np.std(v)))
    for i in range(p.n_fc_layers):
        v = session.run(fc_w[i])
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
    for i in range(p.n_fc_layers):
        fc = tf.matmul(layer_value, fc_w[i])
        bn = batch_normalization('fc_bn' + str(i), fc, is_training=is_training, decay=0.9)
        relu = tf.nn.relu(bn)
        fc_dropout = tf.nn.dropout(relu, keep_prob)
        layer_value = fc_dropout
    # Last Fully Connected Layer
    final_output = tf.add(tf.matmul(layer_value, fc_w[-1]), fc_b)
    return final_output


train_output = fusion_network(train_input, p.train_batch_size, is_training=True)
eval_output = fusion_network(eval_input, p.eval_batch_size, is_training=False)
test_output = fusion_network(test_input, p.test_batch_size, is_training=False)

train_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_output, labels=train_target))

train_logits = tf.nn.softmax(train_output)
eval_logits = tf.nn.softmax(eval_output)
test_logits = tf.nn.softmax(test_output)

train_preds = tf.argmax(train_logits, axis=1)
eval_preds = tf.argmax(eval_logits, axis=1)
test_preds = tf.argmax(test_logits, axis=1)


def confusion_matrix(predictions, results, comment='', do_print=True):
    assert predictions.shape[0] == results.shape[0], 'invalid input shape'
    matrix = np.zeros((p.target_size, p.target_size), dtype=np.int32)
    for ii in range(predictions.shape[0]):
        matrix[int(results[ii])][int(predictions[ii])] += 1
    if do_print:
        str_representation = comment + '\n' + np.array2string(matrix)
        log.get().info(str_representation)
    return matrix


def get_accuracy(conf_matrix):
    return 100.0 * np.trace(conf_matrix) / np.sum(conf_matrix)


optimizer = tf.train.MomentumOptimizer(p.learning_rate, 0.93)
train_step = optimizer.minimize(train_loss)

# Initialize Variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saved_train_loss = []
saved_train_acc = []
saved_eval_acc = []
saved_test_acc = []


def get_h5_data(source, indices):
    return list(map(lambda idx: np.expand_dims(source[idx], 3), indices))


def get_h5_labels(source, indices):
    return list(map(lambda idx: source[idx], indices))


for i in range(p.generations):
    # ---------- CV reshuffle if needed ----------
    if (i + 1) % p.cv_reshuffle_every == 0:
        log.get().info('Performing cross-validation reshuffling')
        train_idx, eval_idx = cross_validation_reshuffle(train_data[0])
    # ---------- train ----------
    r_idx = np.random.choice(train_idx, size=p.train_batch_size)
    train_x = tuple(np.asarray(get_h5_data(j, r_idx)) for j in train_data)
    train_y = np.asarray(get_h5_labels(train_labels, r_idx))
    train_dict = {train_input: train_x, train_target: train_y, keep_prob: p.dropout_train}
    sess.run(train_step, feed_dict=train_dict)
    temp_train_loss, temp_train_preds = sess.run([train_loss, train_preds], feed_dict=train_dict)
    # ---------- print weights ----------
    if (i + 1) % p.print_weights_every == 0:
        print_weights(sess)
    # ---------- evaluate ----------
    if (i + 1) % p.eval_every == 0:
        # --- calculate accuracy for train set ---
        train_cm = confusion_matrix(temp_train_preds, train_y, 'confusion matrix for train set', do_print=False)
        temp_train_acc = get_accuracy(train_cm)
        # --- calculate accuracy on validation set ---
        preds = np.zeros(eval_idx.size, np.float32)
        targets = np.zeros(eval_idx.size, np.float32)
        for l in range(0, eval_idx.size, p.eval_batch_size):
            eval_x = tuple(np.asarray(get_h5_data(j, (eval_idx[l],))) for j in train_data)
            eval_y = get_h5_labels(train_labels, (eval_idx[l],))
            _preds = sess.run(eval_preds, feed_dict={eval_input: eval_x, eval_target: eval_y, keep_prob: p.dropout_eval})
            targets[l * p.eval_batch_size:(l + 1) * p.eval_batch_size] = eval_y
            preds[l * p.eval_batch_size:(l + 1) * p.eval_batch_size] = _preds
        eval_cm = confusion_matrix(preds, targets, 'Confusion matrix for evaluation set', do_print=True)
        temp_eval_acc = get_accuracy(eval_cm)
        # --- calculate accuracy on test set ---
        preds = np.zeros(test_data[0].shape[0], np.float32)
        targets = np.zeros(test_data[0].shape[0], np.float32)
        for l in range(0, test_data[0].shape[0], p.test_batch_size):
            test_x = tuple(np.asarray(get_h5_data(j, (l,))) for j in test_data)
            test_y = get_h5_labels(test_labels, (l,))
            _preds = sess.run(test_preds, feed_dict={test_input: test_x, test_target: test_y, keep_prob: p.dropout_test})
            targets[l * p.test_batch_size:(l + 1) * p.test_batch_size] = test_y
            preds[l * p.test_batch_size:(l + 1) * p.test_batch_size] = _preds
        test_cm = confusion_matrix(preds, targets, 'Confusion matrix for test set', do_print=True)
        temp_test_acc = get_accuracy(test_cm)
        # --- record and print results ---
        saved_train_loss.append(temp_train_loss)
        saved_train_acc.append(temp_train_acc)
        saved_eval_acc.append(temp_eval_acc)
        saved_test_acc.append(temp_test_acc)
        acc_and_loss = [(i + 1), temp_train_loss, temp_train_acc, temp_eval_acc, temp_test_acc]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        log.get().info(
            'Generation # {}. Train loss: {:.5f}. Train acc (Eval acc, Test acc): {:.2f} ({:.2f}, {:.2f})'.format(
                *acc_and_loss)
        )

eval_indices = range(0, p.generations, p.eval_every)
# Plot loss over time
plt.plot(eval_indices, saved_train_loss, 'k-')
plt.title('Softmax loss per generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
# plt.show()
plt.savefig('./plots/' + experiment_name + '_loss.png')
# Plot train and eval accuracy
plt.plot(eval_indices, saved_train_acc, 'k-', label='Train set accuracy')
plt.plot(eval_indices, saved_eval_acc, 'g-', label='Validation set accuracy')
plt.plot(eval_indices, saved_test_acc, 'r--', label='Test set accuracy')
plt.title('Train, validation and test accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
# plt.show()
plt.savefig('./plots/' + experiment_name + '_accuracy.png')

train_eval_h5.close()
test_h5.close()

log.get().info("\nDONE!")