import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import ex_config as cfg
import h5py

adni_root = 'C:/dev/ADNI_Multimodal/dataset/'

train_eval_h5 = h5py.File(cfg.h5_cache_dir + '/rois_10/alz_train_eval.h5', 'r')
test_h5 = h5py.File(cfg.h5_cache_dir + '/rois_10/alz_test.h5', 'r')

train_eval_data_smri = train_eval_h5['data/smri_L_ext']
# train_eval_data_md = train_eval_h5['data/smri2']
train_eval_data_md = train_eval_h5['data/smri_R_ext']
train_eval_labels = train_eval_h5['labels']
test_data_smri = test_h5['data/smri_L_ext']
# test_data_md = test_h5['data/smri2']
test_data_md = test_h5['data/smri_R_ext']
test_labels = test_h5['labels']


def cross_validation_reshuffle(train_eval_data, prc = 0.9):
    n = train_eval_data.shape[0]
    idx = np.asarray(range(n))
    np.random.shuffle(idx)
    m = int(n * prc)
    train_idx = idx[:m]
    eval_idx = idx[m:]
    return train_idx, eval_idx

train_idx, eval_idx = cross_validation_reshuffle(train_eval_data_smri)

train_batch_size = 30
eval_batch_size = 1
test_batch_size = 1
learning_rate = 0.005
target_size = 3
num_channels = 1
generations = 2000
eval_every = 50
cv_reshuffle_every = 500
print_weights_every = 500
cv1_s = 16
cv2_s = 32
cv3_s = 64
pool1_s = 2
pool2_s = 2
pool3_s = 2
fc1_s = 16
dropout_train = 0.5
dropout_eval = 1.0
dropout_test = 1.0

img_shape_smri = train_eval_data_smri[0].shape
img_shape_md = train_eval_data_md[0].shape
train_input_shape_smri = [train_batch_size] + list(img_shape_smri) + [1]
train_input_shape_md = [train_batch_size] + list(img_shape_md) + [1]
eval_input_shape_smri = [eval_batch_size] + list(img_shape_smri) + [1]
eval_input_shape_md = [eval_batch_size] + list(img_shape_md) + [1]
test_input_shape_smri = [test_batch_size] + list(img_shape_smri) + [1]
test_input_shape_md = [test_batch_size] + list(img_shape_md) + [1]

train_input_smri = tf.placeholder(tf.float32, shape=train_input_shape_smri)
train_input_md = tf.placeholder(tf.float32, shape=train_input_shape_md)
train_target = tf.placeholder(tf.int32, shape=train_batch_size)
eval_input_smri = tf.placeholder(tf.float32, shape=eval_input_shape_smri)
eval_input_md = tf.placeholder(tf.float32, shape=eval_input_shape_md)
eval_target = tf.placeholder(tf.int32, shape=eval_batch_size)
test_input_smri = tf.placeholder(tf.float32, shape=test_input_shape_smri)
test_input_md = tf.placeholder(tf.float32, shape=test_input_shape_md)
test_target = tf.placeholder(tf.int32, shape=test_batch_size)
keep_prob = tf.placeholder(tf.float32)

smri_cv1_w = tf.get_variable("smri_W1", shape=[5, 5, 5, 1, cv1_s], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
smri_cv2_w = tf.get_variable("smri_W2", shape=[4, 4, 4, cv1_s, cv2_s], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
smri_cv3_w = tf.get_variable("smri_W3", shape=[3, 3, 3, cv2_s, cv3_s], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
# cv1_b = tf.get_variable("b1", shape=[cv1_s], dtype=tf. float32, initializer=tf.contrib.layers.xavier_initializer())
# cv2_b = tf.get_variable("b2", shape=[cv2_s], dtype=tf. float32, initializer=tf.contrib.layers.xavier_initializer())
# cv3_b = tf.get_variable("b3", shape=[cv3_s], dtype=tf. float32, initializer=tf.contrib.layers.xavier_initializer())

md_cv1_w = tf.get_variable("md_W1", shape=[5, 5, 5, 1, cv1_s], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
md_cv2_w = tf.get_variable("md_W2", shape=[4, 4, 4, cv1_s, cv2_s], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
md_cv3_w = tf.get_variable("md_W3", shape=[3, 3, 3, cv2_s, cv3_s], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())


def calc_size_after_pooling(src_3d_shape, maxpool_list):
    s = src_3d_shape
    for mp in maxpool_list:
        s = math.ceil(s[0] / mp),  math.ceil(s[1] / mp), math.ceil(s[2] / mp)
    return s[0] * s[1] * s[2]

fc1_in_size = calc_size_after_pooling((train_input_shape_smri[1], train_input_shape_smri[2], train_input_shape_smri[3]), (pool1_s, pool2_s, pool3_s)) * cv3_s * 2

fc1_w = tf.get_variable("fW1", shape=[fc1_in_size, fc1_s], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
fc1_b = tf.get_variable("fb1", shape=[fc1_s], dtype=tf. float32, initializer=tf.contrib.layers.xavier_initializer())
fc2_w = tf.get_variable("fW2", shape=[fc1_s, target_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
fc2_b = tf.get_variable("fb2", shape=[target_size], dtype=tf. float32, initializer=tf.contrib.layers.xavier_initializer())

def print_weights(sess):
    print('weights and biases values:')
    # for i in (cv1_w, cv1_b, cv2_w, cv2_b, cv3_w, cv3_b, fc1_w, fc1_b, fc2_w, fc2_b):
    for i in (smri_cv1_w, smri_cv2_w, smri_cv3_w, fc1_w, fc1_b, fc2_w, fc2_b):
        v = sess.run(i)
        print('\t mean: %f, std: %f' % (np.asscalar(np.mean(v)), np.asscalar(np.std(v))))

def batch_norm_wrapper(name, inputs, is_training, decay = 0.9):
    epsilon = 1e-3

    with tf.variable_scope('BN') as scope:
        try:
            scale = tf.get_variable(name + '_s', shape=inputs.get_shape()[1:], dtype=tf.float32, initializer=tf.ones_initializer())
            beta = tf.get_variable(name + '_b', shape=inputs.get_shape()[1:], dtype=tf.float32, initializer=tf.zeros_initializer())
            mean = tf.get_variable(name + '_m', shape=inputs.get_shape()[1:], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)
            var = tf.get_variable(name + '_v', shape=inputs.get_shape()[1:], dtype=tf.float32, initializer=tf.ones_initializer(), trainable=False)
        except ValueError:
            scope.reuse_variables()
            scale = tf.get_variable(name + '_s', shape=inputs.get_shape()[1:], dtype=tf.float32, initializer=tf.ones_initializer())
            beta = tf.get_variable(name + '_b', shape=inputs.get_shape()[1:], dtype=tf.float32, initializer=tf.zeros_initializer())
            mean = tf.get_variable(name + '_m', shape=inputs.get_shape()[1:], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)
            var = tf.get_variable(name + '_v', shape=inputs.get_shape()[1:], dtype=tf.float32, initializer=tf.ones_initializer(), trainable=False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
            train_mean = tf.assign(mean, mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(var, var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs, mean, var, beta, scale, epsilon)


def smri_network(smri_input_data, batch_size, is_training):
    # First SMRI Conv-ReLU-MaxPool Layer
    # conv1 = tf.nn.bias_add(tf.nn.conv3d(input_data, cv1_w, strides=[1, 1, 1, 1, 1], padding='SAME'), cv1_b)
    smri_conv1 = tf.nn.conv3d(smri_input_data, smri_cv1_w, strides=[1, 1, 1, 1, 1], padding='SAME')
    smri_bn1 = batch_norm_wrapper("smri_bn1", smri_conv1, is_training = is_training, decay = 0.9)
    smri_relu1 = tf.nn.relu(smri_bn1)
    smri_pool1 = tf.nn.max_pool3d(smri_relu1, ksize=[1, pool1_s, pool1_s, pool1_s, 1], strides=[1, pool1_s, pool1_s, pool1_s, 1], padding='SAME')
    # Second Conv-ReLU-MaxPool Layer
    # conv2 = tf.nn.bias_add(tf.nn.conv3d(pool1, cv2_w, strides=[1, 1, 1, 1, 1], padding='SAME'), cv2_b)
    smri_conv2 = tf.nn.conv3d(smri_pool1, smri_cv2_w, strides=[1, 1, 1, 1, 1], padding='SAME')
    smri_bn2 = batch_norm_wrapper("smri_bn2", smri_conv2, is_training = is_training, decay = 0.9)
    smri_relu2 = tf.nn.relu(smri_bn2)
    smri_pool2 = tf.nn.max_pool3d(smri_relu2, ksize=[1, pool2_s, pool2_s, pool2_s, 1], strides=[1, pool2_s, pool2_s, pool2_s, 1], padding='SAME')
    # Third Conv-ReLU-MaxPool Layer
    # conv3 = tf.nn.bias_add(tf.nn.conv3d(pool2, cv3_w, strides=[1, 1, 1, 1, 1], padding='SAME'), cv3_b)
    smri_conv3 = tf.nn.conv3d(smri_pool2, smri_cv3_w, strides=[1, 1, 1, 1, 1], padding='SAME')
    smri_bn3 = batch_norm_wrapper("smri_bn3", smri_conv3, is_training = is_training, decay = 0.9)
    smri_relu3 = tf.nn.relu(smri_bn3)
    smri_pool3 = tf.nn.avg_pool3d(smri_relu3, ksize=[1, pool3_s, pool3_s, pool3_s, 1], strides=[1, pool3_s, pool3_s, pool3_s, 1], padding='SAME')
    # Transform Output into a 1xN layer for next fully connected layer
    flat_output = tf.reshape(smri_pool3, [batch_size, -1])

    # First Fully Connected Layer
    # fc1 = tf.add(tf.matmul(flat_output, fc1_w), fc1_b)
    fc1 = tf.matmul(flat_output, fc1_w)
    bn_fc1 = batch_norm_wrapper("bn_fc1", fc1, is_training = is_training, decay = 0.9)
    fc1_relu = tf.nn.relu(bn_fc1)
    fc1_dropout = tf.nn.dropout(fc1_relu, keep_prob)
    # Second Fully Connected Layer
    final_model_output = tf.add(tf.matmul(fc1_dropout, fc2_w), fc2_b)
    return final_model_output


def fusion_network(smri_input_data, md_input_data, batch_size, is_training):
    # First SMRI Conv-ReLU-MaxPool Layer
    smri_conv1 = tf.nn.conv3d(smri_input_data, smri_cv1_w, strides=[1, 1, 1, 1, 1], padding='SAME')
    smri_bn1 = batch_norm_wrapper("smri_bn1", smri_conv1, is_training=is_training, decay=0.9)
    smri_relu1 = tf.nn.relu(smri_bn1)
    smri_pool1 = tf.nn.max_pool3d(smri_relu1, ksize=[1, pool1_s, pool1_s, pool1_s, 1], strides=[1, pool1_s, pool1_s, pool1_s, 1], padding='SAME')
    # First MD Conv-ReLU-MaxPool Layer
    md_conv1 = tf.nn.conv3d(md_input_data, md_cv1_w, strides=[1, 1, 1, 1, 1], padding='SAME')
    md_bn1 = batch_norm_wrapper("md_bn1", md_conv1, is_training=is_training, decay=0.9)
    md_relu1 = tf.nn.relu(md_bn1)
    md_pool1 = tf.nn.max_pool3d(md_relu1, ksize=[1, pool1_s, pool1_s, pool1_s, 1], strides=[1, pool1_s, pool1_s, pool1_s, 1], padding='SAME')
    # Second SMRI Conv-ReLU-MaxPool Layer
    smri_conv2 = tf.nn.conv3d(smri_pool1, smri_cv2_w, strides=[1, 1, 1, 1, 1], padding='SAME')
    smri_bn2 = batch_norm_wrapper("smri_bn2", smri_conv2, is_training=is_training, decay=0.9)
    smri_relu2 = tf.nn.relu(smri_bn2)
    smri_pool2 = tf.nn.max_pool3d(smri_relu2, ksize=[1, pool2_s, pool2_s, pool2_s, 1], strides=[1, pool2_s, pool2_s, pool2_s, 1], padding='SAME')
    # Second MD Conv-ReLU-MaxPool Layer
    md_conv2 = tf.nn.conv3d(md_pool1, md_cv2_w, strides=[1, 1, 1, 1, 1], padding='SAME')
    md_bn2 = batch_norm_wrapper("md_bn2", md_conv2, is_training=is_training, decay=0.9)
    md_relu2 = tf.nn.relu(md_bn2)
    md_pool2 = tf.nn.max_pool3d(md_relu2, ksize=[1, pool2_s, pool2_s, pool2_s, 1], strides=[1, pool2_s, pool2_s, pool2_s, 1], padding='SAME')
    # Third SMRI Conv-ReLU-MaxPool Layer
    smri_conv3 = tf.nn.conv3d(smri_pool2, smri_cv3_w, strides=[1, 1, 1, 1, 1], padding='SAME')
    smri_bn3 = batch_norm_wrapper("smri_bn3", smri_conv3, is_training=is_training, decay=0.9)
    smri_relu3 = tf.nn.relu(smri_bn3)
    smri_pool3 = tf.nn.avg_pool3d(smri_relu3, ksize=[1, pool3_s, pool3_s, pool3_s, 1], strides=[1, pool3_s, pool3_s, pool3_s, 1], padding='SAME')
    # Third MD Conv-ReLU-MaxPool Layer
    md_conv3 = tf.nn.conv3d(md_pool2, md_cv3_w, strides=[1, 1, 1, 1, 1], padding='SAME')
    md_bn3 = batch_norm_wrapper("md_bn3", md_conv3, is_training=is_training, decay=0.9)
    md_relu3 = tf.nn.relu(md_bn3)
    md_pool3 = tf.nn.avg_pool3d(md_relu3, ksize=[1, pool3_s, pool3_s, pool3_s, 1], strides=[1, pool3_s, pool3_s, pool3_s, 1], padding='SAME')

    # Transform Output into a 2xN layer for next fully connected layer
    smri_flat = tf.reshape(smri_pool3, [batch_size, -1])
    md_flat = tf.reshape(md_pool3, [batch_size, -1])
    fusion_flat = tf.concat([smri_flat, md_flat], 1)

    # First Fully Connected Layer
    # fc1 = tf.add(tf.matmul(flat_output, fc1_w), fc1_b)
    fc1 = tf.matmul(fusion_flat, fc1_w)
    bn_fc1 = batch_norm_wrapper("fc_bn1", fc1, is_training=is_training, decay=0.9)
    fc1_relu = tf.nn.relu(bn_fc1)
    fc1_dropout = tf.nn.dropout(fc1_relu, keep_prob)
    # Second Fully Connected Layer
    final_model_output = tf.add(tf.matmul(fc1_dropout, fc2_w), fc2_b)
    return final_model_output

# train_output = smri_network(train_input_smri, train_batch_size, is_training=True)
# eval_output = smri_network(eval_input_smri, eval_batch_size, is_training=False)
# test_output = smri_network(test_input_smri, test_batch_size, is_training=False)

train_output = fusion_network(train_input_smri, train_input_md, train_batch_size, is_training=True)
eval_output = fusion_network(eval_input_smri, eval_input_md, eval_batch_size, is_training=False)
test_output = fusion_network(test_input_smri, test_input_md, test_batch_size, is_training=False)

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
        train_idx, eval_idx = cross_validation_reshuffle(train_eval_data_smri)
    # ---------- train ----------
    r_idx = np.random.choice(train_idx, size=train_batch_size)
    train_smri = np.asarray(get_h5_data(train_eval_data_smri, r_idx))
    train_md = np.asarray(get_h5_data(train_eval_data_md, r_idx))
    train_y = np.asarray(get_h5_lbls(train_eval_labels, r_idx))
    train_dict = {train_input_smri: train_smri, train_input_md: train_md, train_target: train_y, keep_prob: dropout_train}
    sess.run(train_step, feed_dict=train_dict)
    temp_train_loss, temp_train_preds = sess.run([train_loss, train_preds], feed_dict=train_dict)
    # ---------- print weights ----------
    if (i + 1) % print_weights_every == 0:
        print_weights(sess)
    # ---------- evaluate ----------
    if (i + 1) % eval_every == 0:
        # --- calculate accuracy for train set ---
        train_cm = confusion_matrix(temp_train_preds, train_y, 'confusion matrix for train set', do_print=False)
        temp_train_acc = get_accuracy(train_cm)
        # --- calculate accuracy on validation set ---
        preds = np.zeros(eval_idx.size, np.float32)
        targets = np.zeros(eval_idx.size, np.float32)
        for j in range(0, eval_idx.size, eval_batch_size):
            eval_smri = get_h5_data(train_eval_data_smri, (eval_idx[j],))
            eval_md = get_h5_data(train_eval_data_md, (eval_idx[j],))
            eval_y = get_h5_lbls(train_eval_labels, (eval_idx[j],))
            _preds = sess.run(eval_preds, feed_dict={eval_input_smri: eval_smri, eval_input_md: eval_md, eval_target: eval_y, keep_prob: dropout_eval})
            targets[j*eval_batch_size:(j+1)*eval_batch_size] = eval_y
            preds[j*eval_batch_size:(j+1)*eval_batch_size] = _preds
        eval_cm = confusion_matrix(preds, targets, 'Confusion matrix for evaluation set', do_print=True)
        temp_eval_acc = get_accuracy(eval_cm)
        # --- calculate accuracy on test set ---
        preds = np.zeros(test_data_smri.shape[0], np.float32)
        targets = np.zeros(test_data_smri.shape[0], np.float32)
        for j in range(0, test_data_smri.shape[0], test_batch_size):
            test_smri = get_h5_data(test_data_smri, (j,))
            test_md = get_h5_data(test_data_md, (j,))
            test_y = get_h5_lbls(test_labels, (j,))
            _preds = sess.run(test_preds, feed_dict={test_input_smri: test_smri, test_input_md: test_md, test_target: test_y, keep_prob: dropout_test})
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