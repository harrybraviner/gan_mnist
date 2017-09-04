import tensorflow as tf
import numpy as np
import mnist_data
import matplotlib.pyplot as plt

class SimpleGAN:

    def __init__(self):
        self.training_data = mnist_data.getTrainingSet()

        # Discriminative net
        self.d_w1 = tf.Variable(tf.truncated_normal(shape = [784, 128], mean=0.0, stddev=0.02), name='d_w1')
        self.d_b1 = tf.Variable(tf.zeros(shape=[128]), name='d_b1')
        self.d_w2 = tf.Variable(tf.truncated_normal(shape = [128, 1], mean=0.0, stddev=0.02), name='d_w2')
        self.d_b2 = tf.Variable(tf.zeros(shape=[1]), name='d_b2')
        self.theta_d = [self.d_w1, self.d_b1, self.d_w2, self.d_b2]

        # Generative net 
        self.g_w1 = tf.Variable(tf.truncated_normal(shape = [100, 128], mean=0.0, stddev=0.02), name='g_w1')
        self.g_b1 = tf.Variable(tf.zeros(shape=[128]), name='g_b1')
        self.g_w2 = tf.Variable(tf.truncated_normal(shape = [128, 784], mean=0.0, stddev=0.02), name='g_w2')
        self.g_b2 = tf.Variable(tf.zeros(shape=[784]), name='g_b2')
        self.theta_g = [self.g_w1, self.g_b1, self.g_w2, self.g_b2]

        self.z = tf.placeholder(tf.float32, shape = [None, 100], name='z')
        self.g_h1 = tf.nn.relu(tf.matmul(self.z, self.g_w1) + self.g_b1)
        self.g_log_prob = tf.matmul(self.g_h1, self.g_w2) + self.g_b2
        self.g_sample = tf.nn.sigmoid(self.g_log_prob)

        self.d_h1_fake = tf.nn.relu(tf.matmul(self.g_sample, self.d_w1) + self.d_b1)
        self.d_logit_fake = tf.matmul(self.d_h1_fake, self.d_w2) + self.d_b2
        self.d_prob_fake = tf.nn.sigmoid(self.d_logit_fake)

        self.x = tf.placeholder(tf.float32, shape = [None, 784], name='x')
        self.d_h1_real = tf.nn.relu(tf.matmul(self.x, self.d_w1) + self.d_b1)
        self.d_logit_real = tf.matmul(self.d_h1_real, self.d_w2) + self.d_b2
        self.d_prob_real = tf.nn.sigmoid(self.d_logit_real)

        self.d_loss = -tf.reduce_mean(tf.log(self.d_prob_real) + tf.log(1.0 - self.d_prob_fake))
        self.g_loss = -tf.reduce_mean(tf.log(self.d_prob_fake))

        self.d_solver = tf.train.AdamOptimizer().minimize(self.d_loss, var_list = self.theta_d)
        self.g_solver = tf.train.AdamOptimizer().minimize(self.g_loss, var_list = self.theta_g)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.d_loss_curr = None
        self.g_loss_curr = None

        self.d_loss_history = []
        self.g_loss_history = []

    def random_z_sample(self, size):
        return np.random.uniform(-1.0, +1.0, size=[size, 100])

    def train_both(self, batch_size):
        image_batch, _ = self.training_data.getNextTrainingBatch(batch_size)
        image_batch = np.reshape(image_batch, newshape = [batch_size, 784])

        _, self.d_loss_curr = self.sess.run([self.d_solver, self.d_loss], feed_dict = {self.x : image_batch, self.z : self.random_z_sample(batch_size)})
        _, self.g_loss_curr = self.sess.run([self.g_solver, self.g_loss], feed_dict = {self.z : self.random_z_sample(batch_size)})

        self.d_loss_history += [self.d_loss_curr]
        self.g_loss_history += [self.g_loss_curr]

    def plot_samples(self):
        samples = self.sess.run([self.g_sample], feed_dict = {self.z : self.random_z_sample(9)})
        samples = np.reshape(samples, newshape = [9, 28, 28])

        f, ax = plt.subplots(3, 3)
        for a in f.axes:
            a.axis('off')
        for i in range(3):
            for j in range(3):
                ax[i][j].imshow(samples[3*i + j], interpolation = 'none', cmap = 'gray')

        plt.show()

