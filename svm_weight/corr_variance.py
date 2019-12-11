from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

d5 = 5
d7 = 7
batch_size = 128
img_size = 28
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
digit5 = mnist.train.images[mnist.train.labels==d5]
digit7 = mnist.train.images[mnist.train.labels==d7]

mean5 = np.mean(digit5,0)
mean7 = np.mean(digit7,0)
corr = np.abs(mean7.reshape(28,28)-mean5.reshape(28,28))/2
global_idx = np.argsort(corr.flatten())[::-1]
plt.semilogx(np.sort(corr.flatten())[::-1],'k--')









size = 200
n_times = 200
madry_corr_matrix = []
for i in range(n_times):
	rand = np.random.randint(0,len(digit5),size)
	mean5 = np.mean(digit5[rand],0)
	rand = np.random.randint(0,len(digit7),size)
	mean7 = np.mean(digit7[rand],0)
	corr = np.abs(mean7.reshape(28,28)-mean5.reshape(28,28))/2
	plt.semilogx(corr.flatten()[global_idx],'r.')
	madry_corr_matrix += [corr.flatten()[global_idx]]
np.save('madry_corr_matrix',np.asarray(madry_corr_matrix))
plt.grid('on')
plt.title('madry')


#ours
loc = [13,14,15]#[13]#np.arange(10,18,1)
crop_size = 26

plt.figure()
mean5 = np.mean(digit5,0)
mean7 = np.mean(digit7,0)
mean7_corr = []
mean5_corr = []
for i in loc:
	for j in loc:
		mean7_corr += [mean7.reshape(28,28)[i-crop_size//2:i+crop_size//2,i-crop_size//2:i+crop_size//2]]
		mean5_corr += [mean5.reshape(28,28)[i-crop_size//2:i+crop_size//2,i-crop_size//2:i+crop_size//2]]
corr = np.abs(np.mean(mean7_corr,0)-np.mean(mean5_corr,0))/2
global_idx = np.argsort(corr.flatten())[::-1]
plt.semilogx(np.sort(corr.flatten())[::-1],'k--')

ours_corr_matrix = []
for i in range(n_times):
	rand = np.random.randint(0,len(digit5),size)
	mean5 = np.mean(digit5[rand],0)
	rand = np.random.randint(0,len(digit7),size)
	mean7 = np.mean(digit7[rand],0)

	mean7_corr = []
	mean5_corr = []
	for i in loc:
		for j in loc:
			mean7_corr += [mean7.reshape(28,28)[i-crop_size//2:i+crop_size//2,i-crop_size//2:i+crop_size//2]]
			mean5_corr += [mean5.reshape(28,28)[i-crop_size//2:i+crop_size//2,i-crop_size//2:i+crop_size//2]]
	corr = np.abs(np.mean(mean7_corr,0)-np.mean(mean5_corr,0))/2
	plt.semilogx(corr.flatten()[global_idx],'r.')
	ours_corr_matrix += [corr.flatten()[global_idx]]
np.save('ours_corr_matrix',np.asarray(ours_corr_matrix))
plt.grid('on')
plt.title('proposed')






plt.figure()
import numpy as np
a=np.load('madry_corr_matrix.npy')
b=np.load('ours_corr_matrix.npy')
bb = []
for i in range(b.shape[1]):
	bb += [np.std(b[:,i])]
aa = []
for i in range(a.shape[1]):
	aa += [np.std(a[:,i])]

import matplotlib.pyplot as plt
plt.semilogx(np.asarray(bb), label='ours')
plt.semilogx(np.asarray(aa), label='madry')
plt.grid('on')
plt.legend()
plt.show()

