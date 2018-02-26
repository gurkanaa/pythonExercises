import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

#display
'''
klm=mnist.train.images
imgplot=plt.imshow(np.reshape(klm[150,:],[28,28]),cmap='gray')
plt.show()
'''

#linear_model
x=
