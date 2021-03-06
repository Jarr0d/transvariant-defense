import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
d5 = 5
d7 = 7
digit5 = mnist.train.images[mnist.train.labels==d5]
#digit5 = digit5*2-1
digit7 = mnist.train.images[mnist.train.labels==d7]
#digit7 = digit7*2-1

mean5 = np.mean(digit5,0)
mean7 = np.mean(digit7,0)
corr = np.abs(mean7.reshape(28,28)-mean5.reshape(28,28))/2
# plt.semilogx(np.sort(corr.flatten())[::-1],'k--')
global_idx = np.argsort(corr.flatten())[::-1]

score = np.zeros(len(global_idx))
threshhold = 30

for i in range(10):
	i += 1
	w = np.load('./0.1/w_{}.npy'.format(i))
	w = abs(w)
	w_idx = np.argsort(-w)
	score[w_idx[:threshhold]] += 1
	score[w_idx[threshhold:]] -= 1

import matplotlib.pyplot as plt

score = score[global_idx]
positive_data = [x if x>0 else 0 for x in score]
negative_data = [x if x<0 else 0 for x in score]

fig = plt.figure()
# plt.bar(np.arange(len(global_idx)), positive_data, width=1)
# plt.bar(np.arange(len(global_idx)), negative_data, width=1)

plt.semilogx(score)


score = np.zeros(len(global_idx))
threshhold = 30
for i in range(10):
	i += 1
	w = np.load('./0.2/w_{}.npy'.format(i))
	w = abs(w)
	w_idx = np.argsort(-w)
	score[w_idx[:threshhold]] += 1
	score[w_idx[threshhold:]] -= 1
import matplotlib.pyplot as plt
score = score[global_idx]
positive_data = [x if x>0 else 0 for x in score]
negative_data = [x if x<0 else 0 for x in score]
plt.semilogx(score)


plt.show()



