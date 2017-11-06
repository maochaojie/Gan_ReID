from __future__ import division
import os
import random
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import pdb

import io
import layers

class Net(object):
	def __init__(self, model_path, transfer):
		self.model_path = model_path
                self.transfer = transfer
		self.load_layers()
		self.layers = layers.Layers(self.weight_dict)

	def load_layers(self):
		if self.model_path is not None:
		    #print type(np.load(self.model_path, encoding='latin1'))
		    self.weight_dict = np.load(self.model_path, encoding='latin1').item()
		else:
		    self.weight_dict = {}
   
   def save_tfmodel(self, sess, t_vars, save_path):
		#pdb.set_trace()
        data_dict = {}
		for var in t_vars:
            name = var.name
			weights = sess.run(var)
			layer_name = name.split('/')[-2]
			weights_name = name.split('/')[-1].split(':')[0]
			if data_dict.has_key(layer_name):
				w_dict = data_dict[layer_name]
			else:
				w_dict = {}
			w_dict[weights_name] = weights
			data_dict[layer_name] = w_dict
		np.save(save_path, data_dict)
		
	def load_DNet(self, prefix = 'd_'):
		return DNet(self.layers, prefix)
    def load_GNet(self, prefix = 'g_')
        return GNet(self.layers, prefix)

class DNet(object):
	def __init__(self, layers, prefix = 'd_'):
		self.prefix = prefix
		self.layers = layers

	def vggNet(self, input, train=False):
		self.relu1_1 = self.layers._conv_layer(input, "conv1_1")
		self.relu1_2 = self.layers._conv_layer(self.relu1_1, "conv1_2")
		self.pool1 = self.layers._max_pool(self.relu1_2, 'pool1')

		self.relu2_1 = self.layers._conv_layer(self.pool1, "conv2_1")
		self.relu2_2 = self.layers._conv_layer(self.relu2_1, "conv2_2")
		self.pool2 = self.layers._max_pool(self.relu2_2, 'pool2')

		self.relu3_1 = self.layers._conv_layer(self.pool2, "conv3_1")
		self.relu3_2 = self.layers._conv_layer(self.relu3_1, "conv3_2")
		self.relu3_3 = self.layers._conv_layer(self.relu3_2, "conv3_3")
		self.pool3 = self.layers._max_pool(self.relu3_3, 'pool3')

		self.relu4_1 = self.layers._conv_layer(self.pool3, "conv4_1")
		self.relu4_2 = self.layers._conv_layer(self.relu4_1, "conv4_2")
		self.relu4_3 = self.layers._conv_layer(self.relu4_2, "conv4_3")
		self.pool4 = self.layers._max_pool(self.relu4_3, 'pool4')

		self.relu5_1 = self.layers._conv_layer(self.pool4, "conv5_1")
		self.relu5_2 = self.layers._conv_layer(self.relu5_1, "conv5_2")
		self.relu5_3 = self.layers._conv_layer(self.relu5_2, "conv5_3")
		self.pool5 = self.layers._max_pool(self.relu5_3, 'pool5')

		self.fc6 = self.layers._fc_layer(self.pool5, "fc6")
		assert self.fc6.get_shape().as_list()[1:] == [4096]

		self.relu6 = tf.nn.relu(self.fc6)
		if train:
		    self.relu6 = tf.nn.dropout(self.relu6, 0.5)

		self.fc7 = self.layers._fc_layer(self.relu6, "fc7")
		self.relu7 = tf.nn.relu(self.fc7)
		if train:
		    self.relu7 = tf.nn.dropout(self.relu7, 0.5)

		self.fc8 = self.layers._fc_layer(self.relu7, "fc8")
		self.prob = tf.nn.softmax(self.fc8, name="prob")
		return self.prob
	def siameseVggNet(self, input, p_input, train=False):
		#### one net #####
		self.relu1_1 = self.layers._conv_layer(input, "conv1_1", self.prefix + "conv1_1")
		self.relu1_2 = self.layers._conv_layer(self.relu1_1, "conv1_2", self.prefix + "conv1_2")
		self.pool1 = self.layers._max_pool(self.relu1_2, self.prefix+'pool1')

		self.relu2_1 = self.layers._conv_layer(self.pool1, "conv2_1", self.prefix + "conv2_1")
		self.relu2_2 = self.layers._conv_layer(self.relu2_1, "conv2_2", self.prefix + "conv2_2")
		self.pool2 = self.layers._max_pool(self.relu2_2, self.prefix + 'pool2')

		self.relu3_1 = self.layers._conv_layer(self.pool2, "conv3_1", self.prefix + "conv3_1")
		self.relu3_2 = self.layers._conv_layer(self.relu3_1, "conv3_2", self.prefix + "conv3_2")
		self.relu3_3 = self.layers._conv_layer(self.relu3_2, "conv3_3", self.prefix + "conv3_3")
		self.pool3 = self.layers._max_pool(self.relu3_3, 'pool3')

		self.relu4_1 = self.layers._conv_layer(self.pool3, "conv4_1", self.prefix + "conv4_1")
		self.relu4_2 = self.layers._conv_layer(self.relu4_1, "conv4_2", self.prefix + "conv4_2")
		self.relu4_3 = self.layers._conv_layer(self.relu4_2, "conv4_3", self.prefix + "conv4_3")
		self.pool4 = self.layers._max_pool(self.relu4_3, 'pool4')

		self.relu5_1 = self.layers._conv_layer(self.pool4, "conv5_1", self.prefix + "conv5_1")
		self.relu5_2 = self.layers._conv_layer(self.relu5_1, "conv5_2", self.prefix + "conv5_2")
		self.relu5_3 = self.layers._conv_layer(self.relu5_2, "conv5_3", self.prefix + "conv5_3")

		#### p net ######

		self.p_relu1_1 = self.layers._conv_layer(p_input, "conv1_1", self.prefix + "p_conv1_1")
		self.p_relu1_2 = self.layers._conv_layer(self.p_relu1_1, "conv1_2", self.prefix + "p_conv1_2")
		self.p_pool1 = self.layers._max_pool(self.p_relu1_2, self.prefix+'p_pool1')

		self.p_relu2_1 = self.layers._conv_layer(self.p_pool1, "conv2_1", self.prefix + "p_conv2_1")
		self.p_relu2_2 = self.layers._conv_layer(self.p_relu2_1, "conv2_2", self.prefix + "p_conv2_2")
		self.p_pool2 = self.layers._max_pool(self.p_relu2_2, self.prefix + 'p_pool2')

		self.p_relu3_1 = self.layers._conv_layer(self.p_pool2, "conv3_1", self.prefix + "p_conv3_1")
		self.p_relu3_2 = self.layers._conv_layer(self.p_relu3_1, "conv3_2", self.prefix + "p_conv3_2")
		self.p_relu3_3 = self.layers._conv_layer(self.p_relu3_2, "conv3_3", self.prefix + "p_conv3_3")
		self.p_pool3 = self.layers._max_pool(self.p_relu3_3, 'p_pool3')

		self.p_relu4_1 = self.layers._conv_layer(self.p_pool3, "conv4_1", self.prefix + "p_conv4_1")
		self.p_relu4_2 = self.layers._conv_layer(self.p_relu4_1, "conv4_2", self.prefix + "p_conv4_2")
		self.p_relu4_3 = self.layers._conv_layer(self.p_relu4_2, "conv4_3", self.prefix + "p_conv4_3")
		self.p_pool4 = self.layers._max_pool(self.p_relu4_3, 'p_pool4')

		self.p_relu5_1 = self.layers._conv_layer(self.p_pool4, "conv5_1", self.prefix + "p_conv5_1")
		self.p_relu5_2 = self.layers._conv_layer(self.p_relu5_1, "conv5_2", self.prefix + "p_conv5_2")
		self.p_relu5_3 = self.layers._conv_layer(self.p_relu5_2, "conv5_3", self.prefix + "p_conv5_3")

		#### concat ######

		self.concat_1 = tf.concat([self.relu5_3, self.p_relu5_3], axis=3)

		self.conv6_1 = self.layers._conv2d_layer(self.concat_1, 1024, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,name= self.prefix + "conv6_1")
		self.conv6_2 = self.layers._conv2d_layer(self.conv6_1, 1024, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,name= self.prefix + "conv6_2")
		self.pool5 = self.layers._max_pool(self.conv6_2, 'pool5')

		self.fc7 = self.layers._fcNW_layer(self.pool5, 1024, self.prefix + "fc7")
		self.relu7 = tf.nn.relu(self.fc7)
		if train:
		    self.relu7 = tf.nn.dropout(self.relu7, 0.5)

		self.fc8 = self.layers._fcNW_layer(self.relu7, 1024, self.prefix + "fc8")
		self.relu8 = tf.nn.relu(self.fc8)
		if train:
		    self.relu8 = tf.nn.dropout(self.relu8, 0.5)

		self.fc9 = self.layers._fcNW_layer(self.relu8, 2, self.prefix + "fc9")
		self.prob = tf.nn.softmax(self.fc9, name = self.prefix +"prob")

		return self.fc9, self.prob

class GNet(object):
	def __init__(self, layers, prefix = 'g_'):
		self.prefix = prefix
		self.layers = layers
	def uNet(self, input, train=False):
    	# encoder: [batch, 160, 80, in_channels] => [batch, 10, 5, 64*8]
    	self.relu1 = self.layers._conv2d_layer(input, 64, k_h=5, k_w=5, d_h=2, d_w=2, \
    		stddev=0.02,name= self.prefix + "conv1_1")
        self.relu2 = self.layers._conv2d_layer(self.relu1, 64*2, k_h=3, k_w=3, d_h=2, d_w=2, \
        	stddev=0.02,name= self.prefix + "conv2_1")
        self.relu2_bn = self.layers._batchnorm(self.relu2, name= self.prefix + "bn2")
        self.relu3 = self.layers._conv2d_layer(self.relu2_bn, 64*4, k_h=3, k_w=3, d_h=2, d_w=2, \
        	stddev=0.02,name= self.prefix + "conv3_1")
        self.relu3_bn = self.layers._batchnorm(self.relu3, name= self.prefix + "bn3")
        self.relu4 = self.layers._conv2d_layer(self.relu3_bn, 64*8, k_h=3, k_w=3, d_h=2, d_w=2, \
        	stddev=0.02,name= self.prefix + "conv4_1")
        self.relu4_bn = self.layers._batchnorm(self.relu4, name= self.prefix + "bn4")

        # decoder
        self.relu3_de = self.layers._deconv2d_layer(self.relu4_bn, 64*4, k_h=3, k_w=3, d_h=2, d_w=2, \
        	stddev=0.02,name= self.prefix + "deconv3_1")
        self.relu3_de_bn = self.layers._batchnorm(self.relu3_de, name= self.prefix + "de_bn3")
        self.concat3_de = tf.concat([self.relu3_de_bn, self.relu3_bn], axis=3)

        self.relu2_de = self.layers._deconv2d_layer(self.concat3_de, 64*2, k_h=3, k_w=3, d_h=2, d_w=2, \
        	stddev=0.02,name= self.prefix + "deconv2_1")
        self.relu2_de_bn = self.layers._batchnorm(self.relu2_de, name= self.prefix + "de_bn2")
        self.concat2_de = tf.concat([self.relu2_de_bn, self.relu2_bn], axis=3)

        self.relu1_de = self.layers._deconv2d_layer(self.concat2_de, 64, k_h=3, k_w=3, d_h=2, d_w=2, \
        	stddev=0.02,name= self.prefix + "deconv1_1")
        self.relu1_de_bn = self.layers._batchnorm(self.relu1_de, name= self.prefix + "de_bn1")
        self.concat1_de = tf.concat([self.relu1_de_bn, self.relu1_bn], axis=3)

        self.relu_de = self.layers._deconv2d_layer(self.concat1_de, 3, k_h=5, k_w=5, d_h=2, d_w=2, \
        	stddev=0.02,name= self.prefix + "deconv0_1", with_relu = False)
        self.output = tf.tanh(self.relu_de)
        return self.output