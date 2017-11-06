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

import net
import mio

try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter


class solver(object):
	def __init__(self, sess, model_name, phase, train_list, val_list, train_batch_size, val_batch_size, learning_rate, beta1,epoch, \
			model_path = None, input_height = 160, input_width = 80, resize_height = 160, \
			resize_width = 80, crop = False, grayscale = False, sp = ',', img_root = './', \
			checkpoint_dir = './check_point',model_prefix='./tfmodel/', transfer = True, debug = False):
		self.sess = sess
		self.model_name = model_name
		self.phase = phase
		self.train_batch_size = train_batch_size
		self.val_batch_size = val_batch_size
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.epoch = epoch
		
		self.model_path = model_path

		self.input_height = input_height
		self.input_width = input_width
		self.resize_height = resize_height
		self.resize_width = resize_width
		self.crop = crop
		self.grayscale = grayscale
		self.sp = sp
		self.img_root = img_root
		self.checkpoint_dir = checkpoint_dir
		self.model_prefix = model_prefix
		self.DEBUG =debug
		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)
		print self.input_width
		self.ior = mio.IO(self.input_height, self.input_width, \
					resize_height = self.resize_height, resize_width = self.resize_width, \
					crop = self.crop, grayscale = self.grayscale, sp = self.sp, img_root = self.img_root)

		self.train_data = self.ior.read_file(train_list)
		if self.DEBUG:
			print len(self.train_data)
		self.train_batch_idxs = len(self.train_data) // self.train_batch_size
		
		self.val_data = self.ior.read_file(val_list)
		self.val_batch_idxs = len(self.val_data) // self.val_batch_size

		if len(self.train_data) > 0:
			[image1,image2,label] = self.ior.get_image_pair(random.choice(self.train_data))
		elif len(self.val_data) > 0:
			[image1,image2,label] = self.ior.get_image_pair(random.choice(self.val_data))
		self.image_dims = list(image1.shape[:3])
		print self.image_dims

		self.transfer = transfer
		self.net_ = net.Net(self.model_path, self.transfer)
		self.DNet_ = self.net_.load_DNet(prefix='d_')
		self.GNet_ = self.net_.load_GNet(prefix='g_')
		self.build_model()

		


	def build_model(self):
		if self.phase == 'train':
			self.build_model_train()
		elif self.phase == 'test':
			self.build_model_test()
		

		self.t_vars = tf.trainable_variables()

		self.d_vars = [var for var in self.t_vars if 'd_' in var.name]

		self.saver = tf.train.Saver()

	def train(self):
	    #pdb.set_trace()
	    # d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
	              # .minimize(self.d_loss, var_list=self.d_vars)
	    d_optim = tf.train.GradientDescentOptimizer(10e-3).minimize(self.d_loss, var_list=self.d_vars)
	    try:
	      tf.global_variables_initializer().run()
	    except:
	      tf.initialize_all_variables().run()

	    self.writer = SummaryWriter("./logs", self.sess.graph)
	      
	    counter = 1
	    start_time = time.time()
	    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
	    
	    if could_load:
			counter = checkpoint_counter
			print(" [*] Load SUCCESS")
	    else:
	        print(" [!] Load failed...")

	    for epoch in xrange(self.epoch):
	    	random.shuffle(self.train_data)
	    	for idx in xrange(0, self.train_batch_idxs):
				batch_data = self.train_data[idx*self.train_batch_size:(idx+1)*self.train_batch_size]
				[batch_inputs_1, batch_inputs_2, batch_inputs_label] = \
				    self.ior.get_inputs(batch_data)

				# Update D network with real scene
				self.sess.run(d_optim,\
				feed_dict={ self.train_inputs: batch_inputs_1, self.train_semi_inputs: batch_inputs_2, self.train_inputs_label: batch_inputs_label})
				# self.writer.add_summary(summary_str, counter)
				errD = self.d_loss.eval({ self.train_inputs:batch_inputs_1, \
					self.train_semi_inputs: batch_inputs_2, self.train_inputs_label: batch_inputs_label })

				# if counter%100 == 0:
				# pdb.set_trace()
				# print self.t_vars[2].name
				# print self.sess.run(self.t_vars[2])[:10,0]


				counter += 1
				print("Epoch: [%2d] [%4d/%4d] time: %f, d_loss: %f "\
				  % (epoch, idx, self.train_batch_idxs,time.time() - start_time, errD))
				if counter%500 == 0:
					accuracy,avg_loss = self.validation()
					self.save(counter)
					print ("validation: accuracy: %f, d_loss: %f "\
				  	% (accuracy, avg_loss))
				
			self.save_model(counter)
	
	def build_model_train(self):
		## train input
		## input for real flow
		self.train_inputs = tf.placeholder(
		  tf.float32, [self.train_batch_size] + self.image_dims, name='real_images')

		train_inputs = self.train_inputs
		## input for semi_real flow
		self.train_semi_inputs = tf.placeholder(
		  tf.float32, [self.train_batch_size] + self.image_dims, name='semi_images')
		train_semi_inputs = self.train_semi_inputs

		self.train_inputs_label = tf.placeholder(
		  tf.int32, [self.train_batch_size], name='labels')
		train_inputs_label = self.train_inputs_label

		## val input
		## input for real flow
		self.val_inputs = tf.placeholder(
		  tf.float32, [self.val_batch_size] + self.image_dims, name='real_images')

		val_inputs = self.val_inputs
		## input for semi_real flow
		self.val_semi_inputs = tf.placeholder(
		  tf.float32, [self.val_batch_size] + self.image_dims, name='semi_images')
		val_semi_inputs = self.val_semi_inputs

		self.val_inputs_label = tf.placeholder(
		  tf.int32, [self.val_batch_size], name='labels')
		val_inputs_label = self.val_inputs_label

		self.d_net, _  = self.discriminator(train_inputs, train_semi_inputs, reuse=False)
		self.g_net = self.generator(train_inputs, reuse=False)
		self.gd_net, _ = self.discriminator(train_inputs, self.g_net, reuse=False)
		
		self.valor_d, _ = self.valor_discriminator(val_inputs, val_semi_inputs)
		self.valor_g = self.valor_generator(val_inputs)
		self.valor_gd, _ = self.valor_discriminator(val_inputs, self.valor_g)


		self.d_sum = histogram_summary("d", self.d_net)
		self.d_loss = tf.reduce_mean(
		  tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.d_net, labels=train_inputs_label))
		self.g_loss = tf.reduce_mean(
		  tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.gd_net, labels=np.ones((self.train_batch_size)).astype(np.int32)))

		self.val_d_loss = tf.reduce_mean(
		  tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.valor_d, labels=val_inputs_label))
		self.valor_gd_loss = tf.reduce_mean(
		  tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.valor_gd, labels=np.ones((self.val_inputs_label)).astype(np.int32)))
	
	def build_model_test(self):
		## val input
		## input for real flow
		self.test_inputs = tf.placeholder(
		  tf.float32, [self.val_batch_size] + self.image_dims, name='real_images')

		test_inputs = self.test_inputs
		## input for semi_real flow
		self.test_semi_inputs = tf.placeholder(
		  tf.float32, [self.val_batch_size] + self.image_dims, name='semi_images')
		test_semi_inputs = self.test_semi_inputs

		self.test_inputs_label = tf.placeholder(
		  tf.int32, [self.val_batch_size], name='labels')
		test_inputs_label = self.test_inputs_label

		self.testor_d, _ = self.testor_discriminator(test_inputs, test_semi_inputs)
		self.testor_g = self.testor_generator(test_inputs)
		self.testor_gd, _ = self.testor_discriminator(test_inputs, self.testor_g)

		self.testor_sum = histogram_summary("d", self.testor)
		self.test_d_loss = tf.reduce_mean(
		  tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.testor, labels=test_inputs_label))
		self.testor_gd_loss = tf.reduce_mean(
		  tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.testor_gd, labels=np.ones((self.test_inputs_label)).astype(np.int32)))



	def validation(self):
		sum_loss = 0
		right = 0
		for val_idx in xrange(0, self.val_batch_idxs):
			val_data = self.val_data[val_idx*self.val_batch_size:(val_idx+1)*self.val_batch_size]
			[val_inputs_1, val_inputs_2, val_inputs_label] = \
			    self.ior.get_inputs(val_data)

			score,loss = self.sess.run([self.valor,self.val_d_loss], feed_dict={ \
			    self.val_inputs: val_inputs_1, \
			    self.val_semi_inputs:val_inputs_2, \
			    self.val_inputs_label: val_inputs_label \
			}, \
			)
			sum_loss += loss
			right += self.compute_accuracy(score, val_inputs_label)
			accuracy = right*1.0/(self.val_batch_idxs*self.val_batch_size)
			avg_loss = sum_loss/self.val_batch_idxs
		return accuracy,avg_loss 

	def test(self):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		with tf.variable_scope("discriminator") as scope:
			scope.reuse_variables()
		right = 0
		sum_loss = 0
		for val_idx in xrange(0, self.val_batch_idxs):
			val_data = self.val_data[val_idx*self.val_batch_size:(val_idx+1)*self.val_batch_size]
			[val_inputs_1, val_inputs_2, val_inputs_label] = \
			    self.ior.get_inputs(val_data)

			score, loss = self.sess.run([self.testor, self.test_d_loss], feed_dict={ \
			    self.test_inputs: val_inputs_1, \
			    self.test_semi_inputs: val_inputs_2, \
			    self.test_inputs_label: val_inputs_label \
			}, \
			)
			sum_loss += loss
			right += self.compute_accuracy(score, val_inputs_label)
			#print right
		avg_loss = sum_loss/self.val_batch_idxs
		accuracy = right*1.0/(self.val_batch_idxs*self.val_batch_size) 
		print ("test: accuracy: %f, d_loss: %f "\
				  	% (accuracy,avg_loss))


  	def discriminator(self, image, semi_image, reuse = False):
	    with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()
			return self.DNet_.siameseVggNet(image, semi_image, train=True)
			# return self.DNet_.test_net(image, semi_image)

	def generator(self, image, semi_image, reuse = False):
	    with tf.variable_scope("generator") as scope:
			if reuse:
				scope.reuse_variables()
			return self.GNet_.uNet(image, train=True)
			# return self.DNet_.test_net(image, semi_image)
	
	def valor_discriminator(self, image, semi_image):
		with tf.variable_scope("discriminator") as scope:
		    scope.reuse_variables()
		    return self.DNet_.siameseVggNet(image, semi_image, train=False)
		    # return self.DNet_.test_net(image, semi_image)
	def valor_generator(self, image, semi_image):
		with tf.variable_scope("generator") as scope:
		    scope.reuse_variables()
		    return self.GNet_.uNet(image, train=False)
		    # return self.DNet_.test_net(image, semi_image)      
	def testor_discriminator(self, image, semi_image):
		with tf.variable_scope("discriminator") as scope:
		  	return self.DNet_.siameseVggNet(image, semi_image, train=False)

	def testor_generator(self, image, semi_image):
		with tf.variable_scope("generator") as scope:
		  	return self.GNet_.uNet(image, train=False)

	def compute_accuracy(self, score, labels):
		right = 0
		for idx,label in enumerate(labels):
			if (score[idx,1] > score[idx,0]) == (label > 0):
				right += 1
		return right  


	@property
	def model_dir(self):
		return "{}_{}".format(
		    self.model_name, self.train_batch_size)
	  
	def save(self, step):
		checkpoint_dir_path = os.path.join(self.checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir_path):
		  os.makedirs(checkpoint_dir_path)

		self.saver.save(self.sess,
		        os.path.join(checkpoint_dir_path, self.model_name),
		        global_step=step)

        def save_model(self, step):
		model_dir_path = os.path.join(self.model_prefix, self.model_dir+'{}.npy'.format(step))
		if not os.path.exists(self.model_prefix):
			os.makedirs(self.model_prefix)
		self.net_.save_tfmodel(self.sess, self.t_vars, model_dir_path)
	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
		  ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		  self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
		  counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
		  print(" [*] Success to read {}".format(ckpt_name))
		  return True, counter
		else:
		  print(" [*] Failed to find a checkpoint")
		  return False, 0
