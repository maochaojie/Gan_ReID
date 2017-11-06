import tensorflow as tf
import numpy as np
import net

class Layers(object):
    def __init__(self, weight_dict):
        self.weight_dict = weight_dict
    def get_conv_filter(self, name):
        assert self.weight_dict.has_key(name)
        return self.weight_dict[name]['weights']

    def get_bias(self, name):
        assert self.weight_dict.has_key(name)
        return self.weight_dict[name]['biases']

    def get_fc_weight(self, name):
        raise NotImplementedError

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME', name=name)

    def _conv_layer(self, bottom, weightName, name='conv2d', d_h=1, d_w=1, with_relu = True):
        with tf.variable_scope(name) as scope:
            filt = tf.get_variable('weights',
                      initializer=self.get_conv_filter(weightName))
            conv = tf.nn.conv2d(bottom, filt, [1, d_h, d_w, 1], padding='SAME')

            conv_biases = tf.get_variable('biases',
                initializer=self.get_bias(weightName))
            bias = tf.nn.bias_add(conv, conv_biases)
            if with_relu:
                relu = tf.nn.relu(bias)
                return relu
            else:
                return bias
    def _conv2d_layer(self, input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d", with_relu = True):
        with tf.variable_scope(name) as scope:
            filt = tf.get_variable('weights', [k_h, k_w, input_.get_shape()[-1], output_dim],
                      initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, filt, strides=[1, d_h, d_w, 1], padding='SAME')

            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            
            if with_relu:
                relu = tf.nn.relu(conv)
                return relu
            else:
                return conv

    def _deconv_layer(self, input, output_dim, weightName, name='deconv2d', d_h=1, d_w=1, with_relu = True):
        with tf.variable_scope(name) as scope:
            filt = tf.get_variable('weights',
                      initializer=self.get_conv_filter(weightName))
            batch, in_height, in_width, in_channels = [int(d) for d in input_.get_shape()]
            deconv = tf.nn.conv2d_transpose(input_, filt, [batch, in_height * d_h, in_width * d_w, out_channels], \
                    [1, d_h, d_w, 1], padding="SAME")
            if with_relu:
                relu = tf.nn.relu(deconv)
                return relu
            else:
                return deconv


    def _deconv2d_layer(self, input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d",with_relu = True):
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            batch, in_height, in_width, in_channels = [int(d) for d in input_.get_shape()]
            filt = tf.get_variable('w', [k_h, k_w, output_dim, in_channels],\
                initializer=tf.random_normal_initializer(stddev=stddev))

            deconv = tf.nn.conv2d_transpose(input_, filt, [batch, in_height * d_h, in_width * d_w, out_channels] \
                ,strides=[1, d_h, d_w, 1])
            if with_relu:
                relu = tf.nn.relu(deconv)
                return relu
            else:
                return deconv
     


    def _fc_layer(self, bottom, weightName, name):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                 dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(weightName)
            biases = self.get_bias(weightName)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc
    def _fcNW_layer(self, bottom, output_size, name, stddev = 0.001, bias_start = 1):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                 dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = tf.get_variable("weights", [dim, output_size], tf.float32,
                    tf.random_normal_initializer(stddev=stddev))
            biases = tf.get_variable("biases", [output_size],
                    initializer=tf.constant_initializer(bias_start))

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc
    def _batchnorm(self, input, name = "batchnorm"):
        with tf.variable_scope(name):
            # this block looks like it has 3 inputs on the graph unless we do this
            input = tf.identity(input)

            channels = input.get_shape()[3]
            offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
            scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
            variance_epsilon = 1e-5
            normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
            return normalized
            
    

