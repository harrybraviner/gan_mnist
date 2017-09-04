import tensorflow as tf
import mnist_data
import time
from math import ceil
import numpy as np

class ConvGan:

    def __init__(self):
        
        initial_weight_noise = 0.02
        self.z_width = 100
        self.batch_size = 32
        self.training_data = mnist_data.getTrainingSet(1.0)

        rows = self.training_data.image_set.rows
        cols = self.training_data.image_set.cols

        num_pooled_elements = int(rows*cols*32/4)

        # Weights for generative network
        self.gnet_W1 = tf.Variable(tf.truncated_normal(shape = [self.z_width, 128], stddev = initial_weight_noise, dtype = tf.float16))
        self.gnet_b1 = tf.Variable(tf.constant(value = 0.0, shape = [128], dtype = tf.float16))
        self.gnet_W2 = tf.Variable(tf.truncated_normal(shape = [128, 1024], stddev = initial_weight_noise, dtype = tf.float16))
        self.gnet_b2 = tf.Variable(tf.constant(value = 0.0, shape = [1024], dtype = tf.float16))
        # This next layer forms the inputs to the 'deconvolution' step
        self.gnet_W3 = tf.Variable(tf.truncated_normal(shape = [1024, num_pooled_elements], stddev = initial_weight_noise, dtype = tf.float16))
        self.gnet_b3 = tf.Variable(tf.constant(value = 0.0, shape = [num_pooled_elements], dtype = tf.float16))
        # Finally we have the convolutional weights
        self.gnet_W4 = tf.Variable(tf.truncated_normal(shape = [28, 28, 32, 1], stddev = initial_weight_noise, dtype = tf.float16))
        self.gnet_b4 = tf.Variable(tf.constant(value = 0.0, shape = [28, 28], dtype = tf.float16))

        self.theta_g = [self.gnet_W1, self.gnet_b1, \
                        self.gnet_W2, self.gnet_b2, \
                        self.gnet_W3, self.gnet_b3, \
                        self.gnet_W4, self.gnet_b4]
        # Matrix for 'depooling'
        self.gnet_U = tf.constant([[ 1.0 if int(j/2) == i else 0.0 for i in range(int(rows/2))] for j in range(rows)], shape = [rows, int(rows/2)], \
                                 dtype = tf.float16)

        # Connections for generative network
        self.gnet_z = tf.placeholder(dtype = tf.float16, shape = [None, self.z_width])
        # Two fully connected layers
        self.gnet_h1 = tf.nn.relu(tf.matmul(self.gnet_z,  self.gnet_W1) + self.gnet_b1)
        self.gnet_h2 = tf.nn.relu(tf.matmul(self.gnet_h1, self.gnet_W2) + self.gnet_b2)
        # Fully connected layer for input to convolution
        self.gnet_h3 = tf.nn.relu(tf.matmul(self.gnet_h2, self.gnet_W3) + self.gnet_b3)
        self.gnet_h3_image = tf.reshape(self.gnet_h3, shape = [-1, int(rows/2), int(cols/2)])
        # De-pooling
        self.gnet_h3_depooled = tf.reshape(tf.tensordot(tf.tensordot(self.gnet_h3_image, self.gnet_U, axes = [[1], [1]]), self.gnet_U, axes = [[1], [1]]), \
                                           shape = [-1, rows, cols, 32])
        # Convolution - input to final sigmoid
        self.gnet_z4 = tf.nn.conv2d(self.gnet_h3_depooled, self.gnet_W4, strides = [1, 1, 1, 1], padding = 'SAME') + self.gnet_b4
        self.gnet_g = tf.nn.sigmoid(tf.reshape(self.gnet_z4, shape = [-1, rows, cols]))
