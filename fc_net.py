import tensorflow as tf
import numpy as np
import unittest

class FullyConnectedNet:

    def __init__(self, layer_sizes, input_size, output_size, dtype=tf.float32):

        self._layer_sizes = layer_sizes
        self._input_size = input_size
        self._output_size = output_size
        self._dtype = dtype
        if dtype == tf.float32:
            self._npdtype = np.float32
        elif dtype == tf.float16:
            self._npdtype = np.float16
        else:
            raise ValueError('dtype must be either tf.float32 or tf.float16. Got {}'.format(dtype))
        self._trainable_weights = []

        self._built = False
        self.build()

    def build(self):

        input_sizes  = [self._input_size] + self._layer_sizes
        output_sizes = self._layer_sizes + [self._output_size]
        
        self._weights = [tf.Variable(tf.truncated_normal(shape = [n_in, n_out], stddev = np.sqrt(2/n_in, dtype=self._npdtype), dtype=self._dtype))
                         for (n_in, n_out) in zip(input_sizes, output_sizes)]
        self._biases = [tf.Variable(tf.constant(value = np.sqrt(2/n_in, dtype=self._npdtype), shape=[n_out]), dtype=self._dtype)
                        for (n_in, n_out) in zip(input_sizes, output_sizes)]

        self._built = True

    def connect(self, x):

        self._h = [tf.matmul(x, self._weights[0]) + self._biases[0]]
        for (w, b) in zip(self._weights[1:], self._biases[1:]):
            self._h.append(tf.matmul(self._h[-1], w) + b)

        return self._h[-1]

    @property
    def trainable_weights(self):
        return self._weights + self._biases

class FullyConnectedNetTests(unittest.TestCase):

    def _get_tensor_shape(x):
        return [xx.value for xx in x.shape]

    def test_built_variable_sizes(self):

        with tf.variable_scope('test_built_variable_sizes'):
            fc_gen = FullyConnectedNet([7, 10], 5, 40)

            self.assertEqual(len(fc_gen._weights), 3)
            self.assertEqual(len(fc_gen._biases), 3)
            self.assertEqual(FullyConnectedNetTests._get_tensor_shape(fc_gen._weights[0]), [5, 7])
            self.assertEqual(FullyConnectedNetTests._get_tensor_shape(fc_gen._weights[1]), [7, 10])
            self.assertEqual(FullyConnectedNetTests._get_tensor_shape(fc_gen._weights[2]), [10, 40])
            self.assertEqual(FullyConnectedNetTests._get_tensor_shape(fc_gen._biases[0]), [7])
            self.assertEqual(FullyConnectedNetTests._get_tensor_shape(fc_gen._biases[1]), [10])
            self.assertEqual(FullyConnectedNetTests._get_tensor_shape(fc_gen._biases[2]), [40])

    def test_connected_sizes(self):

        with tf.variable_scope('test_connected_sizes'):
            fc_gen = FullyConnectedNet([7, 10], 5, 40, dtype=tf.float16)

            x = tf.placeholder(dtype=tf.float16, shape=[None, 5])
            y = fc_gen.connect(x)

            self.assertEqual(FullyConnectedNetTests._get_tensor_shape(y), [None, 40])

            self.assertEqual(len(fc_gen.trainable_weights), 6) 

