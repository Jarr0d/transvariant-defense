import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image


# settings
data1 = 5
data2 = 7
eps = 0.2
lmd_1 = 0.3
lmd_2 = 0.02
lr = 0.01
batch_size = 128
img_size = 28
crop_size = 26
epochs = 10
data_partion = 1

# data preparation
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
digit5_train = mnist.train.images[mnist.train.labels == data1]    # num: 4987
digit7_train = mnist.train.images[mnist.train.labels == data2]    # num: 5715
digit5_test = mnist.test.images[mnist.test.labels == data1]    # num: 892
digit7_test = mnist.test.images[mnist.test.labels == data2]    # num: 1028
digit5 = np.concatenate((digit5_train, digit5_test), 0)
digit7 = np.concatenate((digit7_train, digit7_test), 0)

x = tf.placeholder(tf.float32, shape=(None, img_size*img_size))
y = tf.placeholder(tf.float32, shape=(None))

# standard training
# w = np.random.randn(img_size**2)
# w0 = w / np.linalg.norm(w, ord=2)
# w = tf.Variable(np.asarray(w0, dtype='float32'), name="weights")
#
# pred = tf.reduce_sum(w*x, 1)
# err = tf.clip_by_value(tf.ones_like(pred)-y*pred, 0, 20*batch_size)
# acc = tf.reduce_mean(tf.cast(tf.equal(y,tf.sign(pred)),tf.float32))
# loss = tf.reduce_mean(err) + lmd_1/2*tf.norm(w, ord=2)

# Madry adv. training
w = np.random.randn(img_size**2)
w0 = w / np.linalg.norm(w, ord=2)
w = tf.Variable(np.asarray(w0, dtype='float32'), name="weights")

pred = tf.reduce_sum(w*x, 1)
err = tf.clip_by_value(tf.ones_like(pred) - y*pred + eps*tf.norm(w, ord=1), 0, 20*batch_size)
acc = tf.reduce_mean(tf.cast(tf.equal(y,tf.sign(pred)),tf.float32))
loss = tf.reduce_mean(err) + lmd_2/2*tf.norm(w, ord=2)

# Ours
# w = np.random.randn(crop_size**2)
# w0 = w / np.linalg.norm(w, ord=2)
# w = tf.Variable(np.asarray(w0, dtype='float32'), name="weights")
#
# loc = [13, 14, 15]    #np.arange(10,18,1)
# x_img = tf.reshape(x, (-1, img_size, img_size))
# pred, err = [], []
# for i in loc:
#     for j in loc:
#         x_ij = tf.reshape(x_img[:, i-crop_size//2:i+crop_size//2, j-crop_size//2:j+crop_size//2],
#                           (-1, crop_size*crop_size))
#         pred_ij = tf.reduce_sum(w*x_ij, 1)
#         err_ij = tf.clip_by_value(tf.ones_like(pred_ij) - y*pred_ij + eps*tf.norm(w, ord=1), 0, crop_size*batch_size)
#         pred += [pred_ij]
#         err += [err_ij]
# err = tf.reduce_mean(err)
# pred = tf.reduce_mean(pred, 0)
# acc = tf.reduce_mean(tf.cast(tf.equal(y,tf.sign(pred)),tf.float32))
# loss = tf.reduce_mean(err) + lmd_2/2*tf.norm(w, ord=2)

# training setting
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
FLAGS = tf.app.flags.FLAGS
tfconfig = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True,
)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
init = tf.global_variables_initializer()
sess.run(init)

# data partion
num_digit5 = len(digit5_train)
num_digit7 = len(digit7_train)
num5 = int(num_digit5 * data_partion)
num7 = int(num_digit7 * data_partion)

order_digit5 = np.arange(num_digit5)
np.random.shuffle(order_digit5)
digit5 = digit5_train[order_digit5[:num5]]

order_digit7 = np.arange(num_digit7)
np.random.shuffle(order_digit7)
digit7 = digit7_train[order_digit7[:num7]]

# training
iters = int(min(len(digit5), len(digit7)) / 64) - 1
for epoch in range(epochs):
    # data shuffle
    digit5 = np.random.permutation(digit5)
    digit7 = np.random.permutation(digit7)

    for iter in range(iters):
        digit5_batch = digit5[64*iter:64*iter+64]
        digit7_batch = digit7[64*iter:64*iter+64]
        x_batch_train = np.concatenate((digit5_batch, digit7_batch), 0)
        y_batch_train = np.array([-1]*64 + [1]*64)
        slice = np.random.permutation(np.arange(batch_size))
        x_batch_train = x_batch_train[slice]
        y_batch_train = y_batch_train[slice]
        _, loss_, acc_ = sess.run([train_op, loss, acc], {x:x_batch_train, y:y_batch_train})
        print('epoch: {}, iter: {}, loss: {}, acc: {}'.format(epoch, iter, loss_, acc_))

# weight visualization
weight = sess.run(w)
weight[abs(weight) < 0.01] = 0
size = int(np.sqrt(weight.size))
weight = weight.reshape(size, size)
# interplote to img_size*img_size
weight = np.array(Image.fromarray(weight).resize((img_size, img_size)))
plt.imshow(weight, cmap='seismic')
plt.show()

# np.save('weight_{}'.format(data_partion), weight)

# ground-truth corr
mean5 = np.mean(digit5, 0)
mean7 = np.mean(digit7, 0)
corr = (mean7.reshape(img_size, img_size) - mean5.reshape(img_size, img_size)) / 2
idx = np.argsort(abs(corr.flatten()))[::-1]
corr[abs(corr) < eps] = 0
num = sum(abs(corr.flatten()) > 0)

# weight = np.load('weight_1.npy')
robust_acc = []
nonrobust_acc = []
for n in range(num):
    ind = idx[:n+1]
    acc = sum(abs(weight.flatten()[ind]) > 0) / sum(abs(corr.flatten()[ind]) > 0)
    robust_acc.append(acc)
for n in range(num, img_size*img_size):
    ind = idx[:n+1]
    acc = sum(abs(weight.flatten()[ind]) == 0) / sum(abs(corr.flatten()[ind]) == 0)
    nonrobust_acc.append(acc)
ln1, = plt.plot(range(num), robust_acc, 'b--')
plt.show()

# weight = np.load('weight_0.5.npy')
# acc = []
# for n in range(num):
#     ind = idx[:n+1]
#     acc.append(sum(abs(weight.flatten()[ind]) > 0) / sum(abs(corr.flatten()[ind]) > 0))
# ln2, = plt.plot(range(num), acc,'r--')
#
# weight = np.load('weight_0.2.npy')
# acc = []
# for n in range(num):
#     ind = idx[:n+1]
#     acc.append(sum(abs(weight.flatten()[ind]) > 0) / sum(abs(corr.flatten()[ind]) > 0))
# ln3, = plt.plot(range(num), acc,'g--')
#
# weight = np.load('weight_0.1.npy')
# acc = []
# for n in range(num):
#     ind = idx[:n+1]
#     acc.append(sum(abs(weight.flatten()[ind]) > 0) / sum(abs(corr.flatten()[ind]) > 0))
# ln4, = plt.plot(range(num), acc,'y--')
#
# plt.legend(handles=[ln1, ln2, ln3, ln4], labels=['portion=1.0', 'portion=0.5', 'portion=0.2', 'portion=0.1'],
#            loc='lower right')
# plt.ylim(0.5, 1.01)
# plt.show()