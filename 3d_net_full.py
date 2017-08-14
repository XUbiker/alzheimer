import tensorflow as tf
import pickle
import numpy as np
import math
import augmentation as augm
import preprocess as pp
import matplotlib.pyplot as plt
import ex_config as cfg
import h5py

adni_root = 'C:/dev/ADNI_Multimodal/dataset/'
sets_path = 'sets.pkl'
crop_params = {'shift': (0, 0, -0.05), 'prc': (0.05, 0.05, 0.05)}
max_augm_params = augm.AugmParams(shift=(2, 2, 2))

label_dict = {'NC': 0, 'MCI': 1, 'AD': 2}

with open(sets_path, 'rb') as f:
    train, valid, test = pickle.load(f)


train_h5 = h5py.File(cfg.h5_cache_dir + train.name + '.h5', 'r')
test_h5 = h5py.File(cfg.h5_cache_dir + test.name + '.h5', 'r')


batch_size = 3
img_shape = pp.full_preprocess(train.items[0], adni_root, np.float, max_augm_params, crop_params=crop_params).shape
input_shape = [batch_size] + list(img_shape) + [1]

learning_rate = 0.005
target_size = 3
num_channels = 1
generations = 150
eval_every = 5
cv1_s = 16
cv2_s = 32
cv3_s = 64
cv4_s = 128
cv5_s = 256
pool1_s = 2
pool2_s = 2
pool3_s = 2
pool4_s = 2
pool5_s = 2
fc1_s = 64

x_input = tf.placeholder(tf.float32, shape=input_shape)
y_target = tf.placeholder(tf.int32, shape=(batch_size))
eval_input = tf.placeholder(tf.float32, shape=input_shape)
eval_target = tf.placeholder(tf.int32, shape=(batch_size))

cv1_w = tf.Variable(tf.truncated_normal([7, 7, 7, 1, cv1_s], stddev=0.1, dtype=tf.float32))
cv1_b = tf.Variable(tf.zeros([cv1_s], dtype=tf. float32))
cv2_w = tf.Variable(tf.truncated_normal([5, 5, 5, cv1_s, cv2_s], stddev=0.1, dtype=tf.float32))
cv2_b = tf.Variable(tf.zeros([cv2_s], dtype=tf. float32))
cv3_w = tf.Variable(tf.truncated_normal([5, 5, 5, cv2_s, cv3_s], stddev=0.1, dtype=tf.float32))
cv3_b = tf.Variable(tf.zeros([cv3_s], dtype=tf. float32))
cv4_w = tf.Variable(tf.truncated_normal([3, 3, 3, cv3_s, cv4_s], stddev=0.1, dtype=tf.float32))
cv4_b = tf.Variable(tf.zeros([cv4_s], dtype=tf. float32))
cv5_w = tf.Variable(tf.truncated_normal([3, 3, 3, cv4_s, cv5_s], stddev=0.1, dtype=tf.float32))
cv5_b = tf.Variable(tf.zeros([cv5_s], dtype=tf. float32))

def calc_size_after_pooling(src_3d_shape, maxpool_list):
    s = src_3d_shape
    for mp in maxpool_list:
        s = math.ceil(s[0] / mp),  math.ceil(s[1] / mp), math.ceil(s[2] / mp)
    return s[0] * s[1] * s[2]

fc1_in_size = calc_size_after_pooling((input_shape[1], input_shape[2], input_shape[3]), (pool1_s, pool2_s, pool3_s, pool4_s, pool5_s)) * cv5_s
# print(fc1_in_size)

fc1_w = tf.Variable(tf.truncated_normal([fc1_in_size, fc1_s], stddev=0.1, dtype=tf.float32))
fc1_b = tf.Variable(tf.truncated_normal([fc1_s], stddev=0.1, dtype=tf.float32))
fc2_w = tf.Variable(tf.truncated_normal([fc1_s, target_size], stddev=0.1, dtype=tf.float32))
fc2_b = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))

def full_3D_network(input_data):
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
    # Fourth Conv-ReLU-MaxPool Layer
    conv4 = tf.nn.conv3d(max_pool3, cv4_w, strides=[1, 1, 1, 1, 1], padding='SAME')
    relu4 = tf.nn.relu(tf.nn.bias_add(conv4, cv4_b))
    max_pool4 = tf.nn.max_pool3d(relu4, ksize=[1, pool4_s, pool4_s, pool4_s, 1], strides=[1, pool4_s, pool4_s, pool4_s, 1], padding='SAME')
    # Fifth Conv-ReLU-MaxPool Layer
    conv5 = tf.nn.conv3d(max_pool4, cv5_w, strides=[1, 1, 1, 1, 1], padding='SAME')
    relu5 = tf.nn.relu(tf.nn.bias_add(conv5, cv5_b))
    max_pool5 = tf.nn.max_pool3d(relu5, ksize=[1, pool5_s, pool5_s, pool5_s, 1], strides=[1, pool5_s, pool5_s, pool5_s, 1], padding='SAME')
    # Transform Output into a 1xN layer for next fully connected layer
    flat_output = tf.reshape(max_pool5, [batch_size, -1])
    # First Fully Connected Layer
    fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, fc1_w), fc1_b))
    # Second Fully Connected Layer
    final_model_output = tf.add(tf.matmul(fully_connected1, fc2_w), fc2_b)
    return final_model_output


model_output = full_3D_network(x_input)
test_model_output = full_3D_network(eval_input)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))
prediction = tf.nn.softmax(model_output)
test_prediction = tf.nn.softmax(test_model_output)

# Create accuracy function
def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))
    return 100. * num_correct/batch_predictions.shape[0]

my_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = my_optimizer.minimize(loss)

# Initialize Variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

train_loss = []
train_acc = []
test_acc = []


get_h5_data = lambda source, indices: list(map(lambda idx: np.expand_dims(source[idx], 3), indices))
get_h5_lbls = lambda source, indices: list(map(lambda idx: source[idx], indices))

for i in range(generations):
    rand_index = np.random.choice(train.size(), size=batch_size)
    train_x = get_h5_data(train_h5['data/smri'], rand_index)
    train_y = get_h5_lbls(train_h5['labels'], rand_index)
    train_dict = {x_input: train_x, y_target: train_y}
    sess.run(train_step, feed_dict=train_dict)
    temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
    temp_train_acc = get_accuracy(temp_train_preds, train_y)
    if (i + 1) % eval_every == 0:
        acc = []
        for j in range(0, test.size(), batch_size):
            eval_index = np.asarray(range(j, j+batch_size))
            eval_x = get_h5_data(test_h5['data/smri'], eval_index)
            eval_y = get_h5_lbls(test_h5['labels'], eval_index)
            test_preds = sess.run(test_prediction, feed_dict={eval_input: eval_x, eval_target: eval_y})
            # print(get_accuracy(test_preds, eval_y))
            acc += [get_accuracy(test_preds, eval_y)]
        temp_test_acc = np.asarray(acc).mean()
        # print('test acc: ', temp_test_acc)

        # eval_index = np.random.choice(test.size(), size=batch_size)
        # # eval_sets = valid_data[eval_index]
        # # eval_x = get_image(eval_sets)
        # eval_x = get_h5_data(test_h5['data/smri'], eval_index)
        # # eval_y = get_label(eval_sets)
        # eval_y = get_h5_lbls(test_h5['labels'], eval_index)
        # test_dict = {eval_input: eval_x, eval_target: eval_y}
        # test_preds = sess.run(test_prediction, feed_dict=test_dict)
        # temp_test_acc = get_accuracy(test_preds, eval_y)

        # Record and print results
        train_loss.append(temp_train_loss)
        train_acc.append(temp_train_acc)
        test_acc.append(temp_test_acc)
        acc_and_loss = [(i + 1), temp_train_loss, temp_train_acc, temp_test_acc]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]

        print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))


eval_indices = range(0, generations, eval_every)
# Plot loss over time
plt.plot(eval_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()
# Plot train and test accuracy
plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


train_h5.close()
test_h5.close()