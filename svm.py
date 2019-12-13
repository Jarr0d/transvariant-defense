import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.examples.tutorials.mnist import input_data
d5 = 5
d7 = 7
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
digit5 = mnist.train.images[mnist.train.labels==d5]
digit7 = mnist.train.images[mnist.train.labels==d7]
mean5 = np.mean(digit5,0)
mean7 = np.mean(digit7,0)
corr = np.abs(mean7.reshape(28,28)-mean5.reshape(28,28))/2
corr[corr<0.2]=0
plt.imshow(corr,cmap='seismic')
plt.show()



import tensorflow as tf
import numpy as np
batch_size = 128
img_size = 28
x = tf.placeholder(tf.float32,shape=(None, img_size*img_size))
y = tf.placeholder(tf.float32,shape=(None))
w = np.random.randn(img_size**2)
w0 = w/np.linalg.norm(w,ord=2)
#w0=np.ones(img_size**2)
w = tf.Variable(np.asarray(w0,dtype='float32'),name="weights")
pred = tf.reduce_sum(w*x,1)
#err = tf.clip_by_value(tf.ones_like(pred)-y*pred,0,20*batch_size)
err = tf.clip_by_value(tf.ones_like(pred)-y*pred+0.2*tf.norm(w,ord=1),0,20*batch_size)
acc = tf.reduce_mean(tf.cast(tf.equal(y,tf.sign(pred)),tf.float32))
k = 0.3
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

digit5 = mnist.train.images[mnist.train.labels==d5]
#digit5 = digit5*2-1
digit7 = mnist.train.images[mnist.train.labels==d7]
#digit7 = digit7*2-1
for i in range(10):
	for itr in range(76):
		x_batch_train = np.concatenate([digit5[64*itr:64*itr+64],digit7[64*itr:64*itr+64]], 0)
		y_batch_train = [-1]*64 + [1]*64
		_, loss_i, acc_i = sess.run([train_op, loss, acc], {x:x_batch_train, y:y_batch_train})
		print(itr, loss_i, acc_i)

digit5 = mnist.test.images[mnist.test.labels==d5]
digit7 = mnist.test.images[mnist.test.labels==d7]
x_test = np.concatenate([digit5,digit7],0)
y_test = [-1]*len(digit5) + [1]*len(digit7)
test_acc = sess.run(acc, {x:x_test, y:y_test})
print('test acc: {}'.format(test_acc))

np.save('corr_madry',corr)
np.save('w_madry',sess.run(w))

np.linalg.norm(sess.run(w))
import matplotlib.pyplot as plt
plt.imshow(sess.run(w).reshape(28,28),cmap='seismic')
plt.show()

mean5 = np.mean(digit5,0)
mean7 = np.mean(digit7,0)
corr = np.abs(mean7.reshape(28,28)-mean5.reshape(28,28))/2
plt.semilogx(np.sort(corr.flatten())[::-1],'k--')
idx = np.argsort(corr.flatten())[::-1]
plt.semilogx(abs(sess.run(w).flatten())[idx],'r')
plt.show()

np.savetxt('madry.txt',abs(sess.run(w).flatten())[idx])


print(np.linalg.norm(sess.run(w)))






