import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# settings
data1 = 5
data2 = 7
eps = 0.2
lmd = 0.3
lr = 0.01
batch_size = 128
img_size = 28
epochs = 10


# building network
x = tf.placeholder(tf.float32, shape=(None, img_size*img_size))
y = tf.placeholder(tf.float32, shape=(None))
# weight init.
w = np.random.randn(img_size**2)
w0 = w / np.linalg.norm(w, ord=2)

w = tf.Variable(np.asarray(w0, dtype='float32'), name="weights")
pred = tf.reduce_sum(w*x, 1)

# standard training
# err = tf.clip_by_value(tf.ones_like(pred)-y*pred, 0, 20*batch_size)
# Madry adv. training
err = tf.clip_by_value(tf.ones_like(pred) - y*pred + eps*tf.norm(w, ord=1), 0, 20*batch_size)

acc = tf.reduce_mean(tf.cast(tf.equal(y,tf.sign(pred)),tf.float32))
loss = tf.reduce_mean(err) + lmd/2*tf.norm(w, ord=2)
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# training setting
FLAGS = tf.app.flags.FLAGS
tfconfig = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True,
)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
init = tf.global_variables_initializer()
sess.run(init)

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
## training starts ###
FLAGS = tf.app.flags.FLAGS
tfconfig = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True,
)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
init = tf.global_variables_initializer()
sess.run(init)

digit5 = mnist.train.images[mnist.train.labels==d5]
#digit5 = digit5*2-1
digit7 = mnist.train.images[mnist.train.labels==d7]
#digit7 = digit7*2-1

mean5 = np.mean(digit5,0)
mean7 = np.mean(digit7,0)
corr = np.abs(mean7.reshape(28,28)-mean5.reshape(28,28))/2
# plt.semilogx(np.sort(corr.flatten())[::-1],'k--')
global_idx = np.argsort(corr.flatten())[::-1]

# =========================================================================================
for i in range(10):
	for itr in range(76):
		x_batch_train = np.concatenate([digit5[64 * itr:64 * itr + 64], digit7[64 * itr:64 * itr + 64]], 0)
		y_batch_train = [-1] * 64 + [1] * 64
		_, loss_i, acc_i = sess.run([train_op, loss, acc], {x: x_batch_train, y: y_batch_train})
		print(itr, loss_i, acc_i)
global_w = sess.run(w)

digit5 = mnist.test.images[mnist.test.labels==d5]
digit7 = mnist.test.images[mnist.test.labels==d7]
x_test = np.concatenate([digit5,digit7],0)
y_test = [-1]*len(digit5) + [1]*len(digit7)
test_acc = sess.run(acc, {x:x_test, y:y_test})
print('test acc: {}'.format(test_acc))
np.save('w_madry',global_w)
# =========================================================================================

data_partion = 0.1
num_digit5 = len(digit5)
num_digit7 = len(digit7)
num_total = int((num_digit5 + num_digit7) * data_partion)

num5 = int(num_total / 2)
num7 = num_total - num5

order_digit5 = np.arange(num_digit5)
np.random.shuffle(order_digit5)
data_digit5 = digit5[order_digit5[:num5]]

order_digit7 = np.arange(num_digit7)
np.random.shuffle(order_digit7)
data_digit7 = digit7[order_digit7[:num7]]

# =========================================================================================
# for i in range(10):
# 	for itr in range(7):
# 		x_batch_train = np.concatenate([data_digit5[64*itr:64*itr+64],data_digit7[64*itr:64*itr+64]], 0)
# 		y_batch_train = [-1]*64 + [1]*64
# 		_, loss_i, acc_i = sess.run([train_op, loss, acc], {x:x_batch_train, y:y_batch_train})
# 		print(itr, loss_i, acc_i)
# w_1 = sess.run(w)
#
# digit5 = mnist.test.images[mnist.test.labels==d5]
# digit7 = mnist.test.images[mnist.test.labels==d7]
# x_test = np.concatenate([digit5,digit7],0)
# y_test = [-1]*len(digit5) + [1]*len(digit7)
# test_acc = sess.run(acc, {x:x_test, y:y_test})
# print('test acc: {}'.format(test_acc))
# np.save('w_10',w_1)
# =========================================================================================

# score = np.zeros(len(global_idx))
# threshhold = 30
#
# for i in range(10):
# 	i += 1
# 	w = np.load('w_{}.npy'.format(i))
# 	w = abs(w)
# 	w_idx = np.argsort(-w)
#
# 	score[w_idx[:threshhold]] += 1
# 	score[w_idx[threshhold:]] -= 1
#
# import matplotlib.pyplot as plt
#
# score = score[global_idx]
# positive_data = [x if x>0 else 0 for x in score]
# negative_data = [x if x<0 else 0 for x in score]
#
# fig = plt.figure()
# # plt.bar(np.arange(len(global_idx)), positive_data, width=1)
# # plt.bar(np.arange(len(global_idx)), negative_data, width=1)
#
# plt.semilogx(score)
# plt.show()





# np.linalg.norm(sess.run(w))
# import matplotlib.pyplot as plt
# plt.imshow(sess.run(w).reshape(28,28),cmap='seismic')
# plt.show()
#
# mean5 = np.mean(digit5,0)
# mean7 = np.mean(digit7,0)
# corr = np.abs(mean7.reshape(28,28)-mean5.reshape(28,28))/2
# plt.semilogx(np.sort(corr.flatten())[::-1],'k--')
# idx = np.argsort(corr.flatten())[::-1]
# plt.semilogx(abs(sess.run(w).flatten())[idx],'r')
# plt.show()
#
# print(np.linalg.norm(sess.run(w)))






