import tensorflow as tf
import mnist_data
import time
from math import ceil
import numpy as np

def makeWeight(shape, stddev):
    init_val = tf.truncated_normal(shape = shape, stddev = stddev)
    return(tf.Variable(init_val))

def makeBias(shape, value):
    init_val = tf.constant(value=value, shape=shape)
    return(tf.Variable(init_val))

def flattenPooledLayers(pooled_layer):
    rows = pooled_layer.shape.dims[1].value
    cols = pooled_layer.shape.dims[2].value
    channels = pooled_layer.shape.dims[3].value
    return(tf.reshape(pooled_layer, shape=[-1, rows*cols*channels]))

class ConvAndPoolLayer:

    def __init__(self, input_channels, size, output_channels, stddev = 0.02, bias = 0.1):
        self.W = makeWeight([size, size, input_channels, output_channels], stddev)
        self.b = makeBias([output_channels], bias)

    def connect(self, input_tensor):
        h_conv = tf.nn.relu(tf.nn.conv2d(input_tensor, self.W, strides=[1,1,1,1], padding='SAME') + self.b)
        return (tf.nn.max_pool(h_conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'))

class TotallyConnectedLayer:
    def __init__(self, input_width, output_width, stddev = 0.02, bias = 0.1):
        self.W = makeWeight(shape = [input_width, output_width], stddev = stddev)
        self.b = makeBias(shape = [output_width], value = bias)

    def connect(self, input_tensor):
        return (tf.nn.relu(tf.matmul(input_tensor, self.W) + self.b))

    def connect_sigmoid(self, input_tensor):
        return (tf.nn.sigmoid(tf.matmul(input_tensor, self.W) + self.b))

class GAN:


    def __init__(self, fraction_to_use = 1.0):

        self.batch_size = 32
        initial_weight_noise = 0.02
        initial_bias = 0.1
        self.training_data = mnist_data.getTrainingSet(fraction_to_use)

        rows = self.training_data.image_set.rows
        cols = self.training_data.image_set.cols

        # Note: need to specify the generative network first since its output
        #       is fed into the discriminative network

        #Parameters for generative network
        self.gnet_z_width = 256
        gnet_fc1_width = 1024
        gnet_fc2_width = 1024

        # Generative network
        self.gnet_z = tf.placeholder(dtype=tf.float32, shape = [self.batch_size, self.gnet_z_width])
        self.gnet_fc_layer1 = TotallyConnectedLayer(input_width = self.gnet_z.shape.dims[1].value, output_width = gnet_fc1_width)
        self.gnet_fc_h1 = self.gnet_fc_layer1.connect(self.gnet_z)
        self.gnet_fc_layer2 = TotallyConnectedLayer(input_width = self.gnet_fc_h1.shape.dims[1].value, output_width = gnet_fc2_width)
        self.gnet_fc_h2 = self.gnet_fc_layer2.connect(self.gnet_fc_h1)
        self.gnet_fc_layer3 = TotallyConnectedLayer(input_width = self.gnet_fc_h2.shape.dims[1].value, output_width = rows*cols, bias = 0.0)
        self.gnet_output_flat = self.gnet_fc_layer3.connect_sigmoid(self.gnet_fc_h2)
        self.gnet_output = tf.reshape(self.gnet_output_flat, shape = [self.batch_size, rows, cols, 1])

        # Genuine images
        self.x_images = tf.placeholder(dtype=tf.float32, shape = [self.batch_size, rows, cols, 1])

        # Parameters for discrimative network
        dnet_c1_size = 5
        dnet_c1_channels = 32
        dnet_c2_size = 5
        dnet_c2_channels = 64
        dnet_fc1_width = 256

        # Discriminative network
        self.dnet_conv_layer1 = ConvAndPoolLayer(1, dnet_c1_size, dnet_c1_channels)
        self.dnet_conv_layer2 = ConvAndPoolLayer(dnet_c1_channels, dnet_c2_size, dnet_c2_channels)
        self.dnet_fc_layer = TotallyConnectedLayer(input_width = dnet_c2_channels*ceil(rows/4)*ceil(cols/4), output_width = dnet_fc1_width)
        self.dnet_W_fc2 = tf.Variable(tf.truncated_normal(shape = [dnet_fc1_width, 1], stddev = initial_weight_noise))
        self.dnet_b_fc2 = tf.Variable(tf.constant(value = initial_bias, shape = [1]))

        # Connections for training discriminative network
        self.dnet_conv_h1 = self.dnet_conv_layer1.connect(tf.concat([self.x_images, self.gnet_output], axis=0))
        self.dnet_conv_h2 = self.dnet_conv_layer2.connect(self.dnet_conv_h1)
        self.dnet_conv_h2_flat = flattenPooledLayers(self.dnet_conv_h2)
        self.dnet_fc_h1 = self.dnet_fc_layer.connect(self.dnet_conv_h2_flat)
        self.dnet_y_hat_logits = tf.matmul(self.dnet_fc_h1, self.dnet_W_fc2) + self.dnet_b_fc2

        self.y_dnet = tf.concat([tf.constant(1.0, shape = [self.batch_size, 1]), tf.constant(0.0, shape = [self.batch_size, 1])], axis=0)
        self.dnet_cross_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels = self.y_dnet, \
                                                                  logits = self.dnet_y_hat_logits)

        self.dnet_correct_on_genuine = tf.greater(self.dnet_y_hat_logits[:self.batch_size], 0.5)
        self.dnet_correct_on_fake = tf.less_equal(self.dnet_y_hat_logits[self.batch_size:], 0.5)
        self.dnet_accuracy_on_genuine = tf.reduce_mean(tf.cast(self.dnet_correct_on_genuine, dtype=tf.float32))
        self.dnet_accuracy_on_fake = tf.reduce_mean(tf.cast(self.dnet_correct_on_fake, dtype=tf.float32))

        # Connections for training generative network
        self.dnet_conv_h1_g = self.dnet_conv_layer1.connect(self.gnet_output)
        self.dnet_conv_h2_g = self.dnet_conv_layer2.connect(self.dnet_conv_h1_g)
        self.dnet_conv_h2_flat_g = flattenPooledLayers(self.dnet_conv_h2_g)
        self.dnet_fc_h1_g = self.dnet_fc_layer.connect(self.dnet_conv_h2_flat_g)
        self.dnet_y_hat_logits_g = tf.matmul(self.dnet_fc_h1_g, self.dnet_W_fc2) + self.dnet_b_fc2

        self.dnet_cross_entropy_g = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.constant(1.0, shape = [self.batch_size, 1]), \
                                                                    logits = self.dnet_y_hat_logits_g)
        

        #self.optimizer = tf.train.GradientDescentOptimizer(1e-2)
        self.optimizer = tf.train.AdamOptimizer(1e-5)
        self.d_variables = [self.dnet_conv_layer1.W, self.dnet_conv_layer1.b, self.dnet_conv_layer2.W, self.dnet_conv_layer2.b, \
                            self.dnet_fc_layer.W, self.dnet_fc_layer.b, self.dnet_W_fc2, self.dnet_b_fc2]
        self.g_variables = [self.gnet_fc_layer1.W, self.gnet_fc_layer1.b, self.gnet_fc_layer2.W, self.gnet_fc_layer2.b, \
                            self.gnet_fc_layer3.W, self.gnet_fc_layer3.b]
        self.d_train = self.optimizer.minimize(self.dnet_cross_entropy, var_list = self.d_variables)
        self.g_train = self.optimizer.minimize(self.dnet_cross_entropy_g, var_list = self.g_variables)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        self.total_training_time = 0.0
        self.total_evaluating_time = 0.0
        self.discriminator_training_time = 0.0
        self.generator_training_time = 0.0
        self.examples_discriminator_trained_on = 0
        self.examples_generator_trained_on = 0
        self.examples_discriminator_trained_on_history = []
        self.examples_generator_trained_on_history = []
        self.time_discriminator_trained_history = []
        self.time_generator_trained_history = []
        self.accuracy_on_genuine_history = []
        self.accuracy_on_generated_history = []
        self.cross_entropy_history = []

    def train_discriminator(self):
        start_time = time.time()
        training_batch = self.training_data.getNextTrainingBatch(self.batch_size)[0]
        generator_noise = np.reshape(np.random.normal(size=self.batch_size*self.gnet_z_width), \
                                     newshape = [self.batch_size, self.gnet_z_width])
        self.d_train.run(feed_dict = {self.x_images : training_batch, self.gnet_z : generator_noise})
        end_time = time.time()
        self.total_training_time += (end_time - start_time)
        self.discriminator_training_time += (end_time - start_time)
        self.examples_discriminator_trained_on += self.batch_size

    def train_generator(self):
        start_time = time.time()
        generator_noise = np.reshape(np.random.normal(size=self.batch_size*self.gnet_z_width), \
                                     newshape = [self.batch_size, self.gnet_z_width])
        self.g_train.run(feed_dict = {self.gnet_z : generator_noise})
        end_time = time.time()
        self.total_training_time += (end_time - start_time)
        self.generator_training_time += (end_time - start_time)
        self.examples_generator_trained_on += self.batch_size

    def train_discriminator_for_one_epoch(self):
        num_batches = ceil(self.training_data.N_train / self.batch_size)
        for i in range(num_batches):
            self.train_discriminator()

    def train_generator_for_one_epoch(self):
        num_batches = ceil(self.training_data.N_train / self.batch_size)
        for i in range(num_batches):
            self.train_generator()

    def estimate_performance(self, num_batches):
        start_time = time.time()
        
        acc_genuine, acc_fake, cross_entropy = 0.0, 0.0, 0.0
        for i in range(num_batches):
            validation_batch = self.training_data.getNextValidationBatch(self.batch_size)[0]
            generator_noise = np.reshape(np.random.normal(size=self.batch_size*self.gnet_z_width), \
                                         newshape = [self.batch_size, self.gnet_z_width])
            genuine_acc = self.dnet_accuracy_on_genuine.eval(feed_dict={self.x_images : validation_batch, self.gnet_z : generator_noise})
            fake_acc = self.dnet_accuracy_on_fake.eval(feed_dict={self.x_images : validation_batch, self.gnet_z : generator_noise})
            cross_entropy += self.dnet_cross_entropy.eval(feed_dict={self.x_images : validation_batch, self.gnet_z : generator_noise})
            acc_genuine += genuine_acc
            acc_fake += fake_acc
        acc_genuine /= num_batches
        acc_fake /= num_batches
        cross_entropy /= num_batches

        self.accuracy_on_genuine_history += [acc_genuine]
        self.accuracy_on_generated_history += [acc_fake]
        self.cross_entropy_history += [cross_entropy]
        self.time_generator_trained_history += [self.generator_training_time]
        self.time_discriminator_trained_history += [self.discriminator_training_time]

        end_time = time.time()
        self.total_evaluating_time += (end_time - start_time)

    def evaluate_performance(self):
        num_batches = ceil(self.training_data.N_validation / self.batch_size)
        self.estimate_performance(num_batches)

    def generateSingleExample(self):
        generator_noise = np.reshape(np.random.normal(size=self.batch_size*self.gnet_z_width), \
                                     newshape = [self.batch_size, self.gnet_z_width])
        gnet_output = self.gnet_output.eval(feed_dict = {self.gnet_z : generator_noise})
        return(gnet_output[0])
