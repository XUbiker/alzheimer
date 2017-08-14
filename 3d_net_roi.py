import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import ex_config as cfg
import h5py

adni_root = 'C:/dev/ADNI_Multimodal/dataset/'

train_eval_h5 = h5py.File(cfg.h5_cache_dir + '/rois/alz_train_eval.h5', 'r')
test_h5 = h5py.File(cfg.h5_cache_dir + '/rois/alz_test.h5', 'r')

train_eval_data = train_eval_h5['data/smri']
train_eval_labels = train_eval_h5['labels']
test_data = test_h5['data/smri']
test_labels = test_h5['labels']

def cross_validation_reshuffle(train_eval_data, prc = 0.9):
    n = train_eval_data.shape[0]
    idx = np.asarray(range(n))
    np.random.shuffle(idx)
    m = int(n * prc)
    train_idx = idx[:m]
    eval_idx = idx[m:]
    return train_idx, eval_idx

train_idx, eval_idx = cross_validation_reshuffle(train_eval_data)

train_batch_size = 10
eval_batch_size = 1
test_batch_size = 1
learning_rate = 0.005
target_size = 3
num_channels = 1
generations = 10000
eval_every = 50
cv_reshuffle_every = 500
cv1_s = 32
cv2_s = 64
cv3_s = 128
pool1_s = 2
pool2_s = 2
pool3_s = 2
fc1_s = 64
dropout_train = 0.5
dropout_eval = 1.0
dropout_test = 1.0

img_shape = train_eval_data[0].shape
train_input_shape = [train_batch_size] + list(img_shape) + [1]
eval_input_shape = [eval_batch_size] + list(img_shape) + [1]
test_input_shape = [test_batch_size] + list(img_shape) + [1]

train_input = tf.placeholder(tf.float32, shape=train_input_shape)
train_target = tf.placeholder(tf.int32, shape=train_batch_size)
eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
eval_target = tf.placeholder(tf.int32, shape=eval_batch_size)
test_input = tf.placeholder(tf.float32, shape=test_input_shape)
test_target = tf.placeholder(tf.int32, shape=test_batch_size)
keep_prob = tf.placeholder(tf.float32)

cv1_w = tf.get_variable("W1", shape=[5, 5, 5, 1, cv1_s], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
cv1_b = tf.get_variable("b1", shape=[cv1_s], dtype=tf. float32, initializer=tf.contrib.layers.xavier_initializer())
cv2_w = tf.get_variable("W2", shape=[4, 4, 4, cv1_s, cv2_s], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
cv2_b = tf.get_variable("b2", shape=[cv2_s], dtype=tf. float32, initializer=tf.contrib.layers.xavier_initializer())
cv3_w = tf.get_variable("W3", shape=[3, 3, 3, cv2_s, cv3_s], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
cv3_b = tf.get_variable("b3", shape=[cv3_s], dtype=tf. float32, initializer=tf.contrib.layers.xavier_initializer())

def calc_size_after_pooling(src_3d_shape, maxpool_list):
    s = src_3d_shape
    for mp in maxpool_list:
        s = math.ceil(s[0] / mp),  math.ceil(s[1] / mp), math.ceil(s[2] / mp)
    return s[0] * s[1] * s[2]

fc1_in_size = calc_size_after_pooling((train_input_shape[1], train_input_shape[2], train_input_shape[3]), (pool1_s, pool2_s, pool3_s)) * cv3_s

fc1_w = tf.get_variable("fW1", shape=[fc1_in_size, fc1_s], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
fc1_b = tf.get_variable("fb1", shape=[fc1_s], dtype=tf. float32, initializer=tf.contrib.layers.xavier_initializer())
fc2_w = tf.get_variable("fW2", shape=[fc1_s, target_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
fc2_b = tf.get_variable("fb2", shape=[target_size], dtype=tf. float32, initializer=tf.contrib.layers.xavier_initializer())

def network(input_data, batch_size):
    # First Conv-ReLU-MaxPool Layer
    conv1 = tf.nn.conv3d(input_data, cv1_w, strides=[1, 1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, cv1_b))
    max_pool1 = tf.nn.max_pool3d(relu1, ksize=[1, pool1_s, pool1_s, pool1_s, 1], strides=[1, pool1_s, pool1_s, pool1_s, 1], padding='SAME')
    # Second Conv-ReLU-MaxPool Layer
    conv2 = tf.nn.conv3d(max_pool1, cv2_w, strides=[1, 1, 1, 1, 1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, cv2_b))
    max_pool2 = tf.nn.max_pool3d(relu2, ksize=[1, pool2_s, pool2_s, pool2_s, 1], strides=[1, pool2_s, pool2_s, pool2_s, 1], padding='SAME')
    # Third Conv-ReLU-MaxPool Layer
    conv3 = tf.nn.conv3d(max_pool2, cv3_w, strides=[1, 1, 1, 1, 1], padding='SAME')
    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, cv3_b))
    max_pool3 = tf.nn.max_pool3d(relu3, ksize=[1, pool3_s, pool3_s, pool3_s, 1], strides=[1, pool3_s, pool3_s, pool3_s, 1], padding='SAME')
    # Transform Output into a 1xN layer for next fully connected layer
    flat_output = tf.reshape(max_pool3, [batch_size, -1])
    # First Fully Connected Layer
    fc1 = tf.add(tf.matmul(flat_output, fc1_w), fc1_b)
    fc1_relu = tf.nn.relu(fc1)
    fc1_dropout = tf.nn.dropout(fc1_relu, keep_prob)
    # Second Fully Connected Layer
    final_model_output = tf.add(tf.matmul(fc1_dropout, fc2_w), fc2_b)
    return final_model_output


train_output = network(train_input, train_batch_size)
eval_output = network(eval_input, eval_batch_size)
test_output = network(test_input, test_batch_size)

train_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_output, labels=train_target))

train_logits = tf.nn.softmax(train_output)
eval_logits = tf.nn.softmax(eval_output)
test_logits = tf.nn.softmax(test_output)

train_preds = tf.argmax(train_logits, axis=1)
eval_preds = tf.argmax(eval_logits, axis=1)
test_preds = tf.argmax(test_logits, axis=1)

def confusion_matrix(predictions, targets, comment ='', do_print = True):
    n = 3
    assert predictions.shape[0] == targets.shape[0], 'invalid input shape'
    matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(predictions.shape[0]):
        matrix[int(targets[i])][int(predictions[i])] += 1
    if do_print:
        print(comment)
        print(matrix)
    return matrix

def get_accuracy(conf_matrix):
    return 100.0 * np.trace(conf_matrix) / np.sum(conf_matrix)

optimizer = tf.train.MomentumOptimizer(learning_rate, 0.93)
train_step = optimizer.minimize(train_loss)

# Initialize Variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saved_train_loss = []
saved_train_acc = []
saved_eval_acc = []
saved_test_acc = []

get_h5_data = lambda source, indices: list(map(lambda idx: np.expand_dims(source[idx], 3), indices))
get_h5_lbls = lambda source, indices: list(map(lambda idx: source[idx], indices))

for i in range(generations):
    # ---------- CV reshuffle if needed ----------
    if (i + 1) % cv_reshuffle_every == 0:
        print('Performing cross-validation reshuffling')
        train_idx, eval_idx = cross_validation_reshuffle(train_eval_data)
    # ---------- train ----------
    r_idx = np.random.choice(train_idx, size=train_batch_size)
    train_x = np.asarray(get_h5_data(train_eval_data, r_idx))
    train_y = np.asarray(get_h5_lbls(train_eval_labels, r_idx))
    train_dict = {train_input: train_x, train_target: train_y, keep_prob: dropout_train}
    sess.run(train_step, feed_dict=train_dict)
    temp_train_loss, temp_train_preds = sess.run([train_loss, train_preds], feed_dict=train_dict)
    # ---------- evaluate ----------
    if (i + 1) % eval_every == 0:
        # --- calculate accuracy for train set ---
        train_cm = confusion_matrix(temp_train_preds, train_y, 'confusion matrix for train set', do_print=False)
        temp_train_acc = get_accuracy(train_cm)
        # --- calculate accuracy on validation set ---
        preds = np.zeros(eval_idx.size, np.float32)
        targets = np.zeros(eval_idx.size, np.float32)
        for j in range(0, eval_idx.size, eval_batch_size):
            eval_x = get_h5_data(train_eval_data, (eval_idx[j],))
            eval_y = get_h5_lbls(train_eval_labels, (eval_idx[j],))
            _preds = sess.run(eval_preds, feed_dict={eval_input: eval_x, eval_target: eval_y, keep_prob: dropout_eval})
            targets[j*eval_batch_size:(j+1)*eval_batch_size] = eval_y
            preds[j*eval_batch_size:(j+1)*eval_batch_size] = _preds
        eval_cm = confusion_matrix(preds, targets, 'Confusion matrix for evaluation set', do_print=True)
        temp_eval_acc = get_accuracy(eval_cm)
        # --- calculate accuracy on test set ---
        preds = np.zeros(test_data.shape[0], np.float32)
        targets = np.zeros(test_data.shape[0], np.float32)
        for j in range(0, test_data.shape[0], test_batch_size):
            test_x = get_h5_data(test_data, (j,))
            test_y = get_h5_lbls(test_labels, (j,))
            _preds = sess.run(test_preds, feed_dict={test_input: test_x, test_target: test_y, keep_prob: dropout_test})
            targets[j*test_batch_size:(j+1)*test_batch_size] = test_y
            preds[j*test_batch_size:(j+1)*test_batch_size] = _preds
        test_cm = confusion_matrix(preds, targets, 'Confusion matrix for test set', do_print=True)
        temp_test_acc = get_accuracy(test_cm)
        # --- record and print results ---
        saved_train_loss.append(temp_train_loss)
        saved_train_acc.append(temp_train_acc)
        saved_eval_acc.append(temp_eval_acc)
        saved_test_acc.append(temp_test_acc)
        acc_and_loss = [(i + 1), temp_train_loss, temp_train_acc, temp_eval_acc, temp_test_acc]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train loss: {:.5f}. Train acc (Eval acc, Test acc): {:.2f} ({:.2f}, {:.2f})'.format(*acc_and_loss))

eval_indices = range(0, generations, eval_every)
# Plot loss over time
plt.plot(eval_indices, saved_train_loss, 'k-')
plt.title('Softmax loss per generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()
# Plot train and eval accuracy
plt.plot(eval_indices, saved_train_acc, 'k-', label='Train set accuracy')
plt.plot(eval_indices, saved_eval_acc, 'g-', label='Validation set accuracy')
plt.plot(eval_indices, saved_test_acc, 'r--', label='Test set accuracy')
plt.title('Train, validation and test accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

train_eval_h5.close()
test_h5.close()