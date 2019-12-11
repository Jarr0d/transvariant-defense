

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
batch_size = 128
img_size = 28
crop_size = 24


mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
digit5 = mnist.train.images[mnist.train.labels==5]
digit7 = mnist.train.images[mnist.train.labels==7]
mean5 = np.mean(digit5,0)
mean7 = np.mean(digit7,0)



x = tf.placeholder(tf.float32,shape=(batch_size, 28**2))
y = tf.placeholder(tf.float32,shape=(batch_size))
w = np.random.randn(crop_size**2)
w0 = w/np.linalg.norm(w,ord=2)
w = tf.Variable(np.asarray(w0,dtype='float32'),name="weights")
loc = [12,13,14,15,16]#[13]#np.arange(10,18,1)
x_img = tf.reshape(x,(batch_size,28,28))
pred = []
err = []
mean7_corr = []
mean5_corr = []
for i in loc:
	for j in loc:
		x_ij = tf.reshape(x_img[:,i-crop_size//2:i+crop_size//2,j-crop_size//2:j+crop_size//2],(batch_size,crop_size*crop_size))
		pred_ij = tf.reduce_sum(w*x_ij,1)
		err_ij = tf.clip_by_value(tf.ones_like(pred_ij)-y*pred_ij+0.2*tf.norm(w,ord=1),0,crop_size*batch_size)
		pred += [pred_ij]
		err += [err_ij]
		mean7_corr += [mean7.reshape(28,28)[i-crop_size//2:i+crop_size//2,i-crop_size//2:i+crop_size//2]]
		mean5_corr += [mean5.reshape(28,28)[i-crop_size//2:i+crop_size//2,i-crop_size//2:i+crop_size//2]]
corr = np.abs(np.mean(mean7_corr,0)-np.mean(mean5_corr,0))/2

err = tf.reduce_mean(err)
pred_vote = tf.reduce_mean(pred,0)
acc = tf.reduce_mean(tf.cast(tf.equal(y,tf.sign(pred_vote)),tf.float32))
k = 0.03
loss = tf.reduce_mean(err)+k/2*tf.norm(w,ord=2)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

from tensorflow.examples.tutorials.mnist import input_data
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

digit5 = mnist.train.images[mnist.train.labels==5]
#digit5 = digit5*2-1
digit7 = mnist.train.images[mnist.train.labels==7]
#digit7 = digit7*2-1
for i in range(10):
	for itr in range(76):
		x_batch_train = np.concatenate([digit5[64*itr:64*itr+64],digit7[64*itr:64*itr+64]], 0)
		y_batch_train = [-1]*64 + [1]*64
		_, loss_i, acc_i = sess.run([train_op, loss, acc], {x:x_batch_train, y:y_batch_train})
		print(itr, loss_i, acc_i)

np.save('corr_ours',corr)
np.save('w_ours',sess.run(w))
plt.imshow(corr,cmap='seismic')
#plt.show()
plt.figure()
plt.semilogx(np.sort(corr.flatten())[::-1],'k--')
idx = np.argsort(corr.flatten())[::-1]
plt.semilogx(abs(sess.run(w).flatten())[idx],'r')
#plt.show()



print(np.linalg.norm(sess.run(w)))
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(sess.run(w).reshape(crop_size,crop_size),cmap='seismic')
plt.show()



