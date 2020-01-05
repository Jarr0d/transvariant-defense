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
lmd_3 = 1.0
lr = 0.01
batch_size = 128
img_size = 28
crop_size = 26
epochs = 5

# plot data correlation
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
digit5 = mnist.train.images[mnist.train.labels == data1]    # num: 4987
digit7 = mnist.train.images[mnist.train.labels == data2]    # num: 5715
mean5 = np.mean(digit5, 0)
mean7 = np.mean(digit7, 0)
corr = (mean7.reshape(img_size, img_size) - mean5.reshape(img_size, img_size)) / 2
corr[abs(corr) < eps] = 0
plt.imshow(corr, cmap='seismic')
plt.show()

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
# w = np.random.randn(img_size**2)
# w0 = w / np.linalg.norm(w, ord=2)
# w = tf.Variable(np.asarray(w0, dtype='float32'), name="weights")
#
# pred = tf.reduce_sum(w*x, 1)
# err = tf.clip_by_value(tf.ones_like(pred) - y*pred + eps*tf.norm(w, ord=1), 0, 20*batch_size)
# acc = tf.reduce_mean(tf.cast(tf.equal(y,tf.sign(pred)),tf.float32))
# loss = tf.reduce_mean(err) + lmd_2/2*tf.norm(w, ord=2)

# TRADES adv. training
w = np.random.randn(img_size**2)
w0 = w / np.linalg.norm(w, ord=2)
w = tf.Variable(np.asarray(w0, dtype='float32'), name="weights")

pred = tf.reduce_sum(w*x, 1)
err = tf.clip_by_value(tf.ones_like(pred) - y*pred, 0, 20*batch_size)
err_reg = tf.clip_by_value(tf.ones_like(pred) - pred*pred + tf.abs(pred)*eps*tf.norm(w, ord=1), 0, 20*batch_size)
acc = tf.reduce_mean(tf.cast(tf.equal(y,tf.sign(pred)),tf.float32))
loss = tf.reduce_mean(err) + lmd_2/2*tf.norm(w, ord=2) + lmd_3*tf.reduce_mean(err_reg)

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

# testing
digit5 = mnist.test.images[mnist.test.labels == data1]
digit7 = mnist.test.images[mnist.test.labels == data2]
x_test = np.concatenate([digit5, digit7], 0)
y_test = [-1]*len(digit5) + [1]*len(digit7)
test_acc = sess.run(acc, {x:x_test, y:y_test})
print('test acc: {}'.format(test_acc))

# np.save('corr_madry', corr)
# np.save('w_madry', sess.run(w))

# weight visualization
weight = sess.run(w)
weight = weight / np.linalg.norm(weight, ord=2)
size = int(np.sqrt(weight.size))
weight = weight.reshape(size, size)
# interplote to img_size*img_size
weight = np.array(Image.fromarray(weight).resize((img_size, img_size)))
weight[abs(weight) < 0.01] = 0
plt.imshow(weight, cmap='seismic')
plt.show()

# weight correlation vs. #feature
corr = (mean7.reshape(img_size, img_size) - mean5.reshape(img_size, img_size))/2

# for standard training
plt.semilogx(np.sort(abs(corr.flatten()))[::-1],'k--')
idx = np.argsort(abs(corr.flatten()))[::-1]
plt.semilogx(abs(weight.flatten())[idx],'r')
plt.show()

# for adv. training
w_at = corr.copy()
w_at[abs(w_at) < eps]=0
w_at[w_at > 0] -= eps
w_at[w_at < 0] += eps
w_at_abs = np.sort(abs(w_at))[::-1]/2.
plt.semilogx(np.sort(abs(corr))[::-1], 'k--')
plt.semilogx(w_at_abs, 'r')
plt.show()

# np.savetxt('madry.txt',abs(sess.run(w).flatten())[idx])

# acc. vs. #feature
digit5 = mnist.train.images[mnist.train.labels == data1]
digit7 = mnist.train.images[mnist.train.labels == data2]
y_5 = np.array([-1]*len(digit5))
y_7 = np.array([1]*len(digit7))

mean5 = np.mean(digit5, 0)
mean7 = np.mean(digit7, 0)
corr = (mean7.reshape(img_size, img_size) - mean5.reshape(img_size, img_size))/2
corr = corr.flatten()
idx = np.argsort(abs(corr))[::-1]

# for standard training
weight = sess.run(w)
w_ = np.zeros_like(corr)
clean_acc = []
for p in range(img_size*img_size):
    w_[idx[p]] = weight[idx[p]]
    acc_5 = y_5 * np.matmul(digit5, w_)
    acc_7 = y_7 * np.matmul(digit7, w_)
    # acc_ = (sum(acc_5>0) + sum(acc_7>=0)) / (len(digit5) + len(digit7))
    if sum(acc_7 > 0) == 0:
        acc_ = sum(acc_5 > 0) / len(digit5)
    else:
        acc_ = (sum(acc_5 > 0) + sum(acc_7 > 0)) / (len(digit5) + len(digit7))
    print('acc: {}'.format(acc_))
    clean_acc.append(acc_)
plt.semilogx(clean_acc,'b--')
plt.show()

np.save('clean_acc_train', clean_acc)

# for adv. training
w_ = np.zeros_like(corr)
robust_acc = []
for p in range(img_size*img_size):
    w_[idx[p]] = corr[idx[p]]
    acc_5 = y_5 * np.matmul(digit5, w_) - eps*np.linalg.norm(w_, ord=1)
    acc_7 = y_7 * np.matmul(digit7, w_) - eps*np.linalg.norm(w_, ord=1)
    if sum(acc_7 > 0) == 0:
        acc_ = sum(acc_5 > 0) / len(digit5)
    else:
        acc_ = (sum(acc_5 > 0) + sum(acc_7 > 0)) / (len(digit5) + len(digit7))
    print('acc: {}'.format(acc_))
    robust_acc.append(acc_)
plt.semilogx(robust_acc,'r--')
plt.show()

np.save('robust_acc_train', robust_acc)
