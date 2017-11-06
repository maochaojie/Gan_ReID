from __future__ import division
import os
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange


class IO(object):
	def __init__(self, input_height, input_width, \
					resize_height=64, resize_width=64, \
					crop=True, grayscale=False, sp=' ', img_root='./'):
		self.input_height = input_height
		self.input_width = input_width
		self.resize_height = resize_height
		self.resize_width = resize_width
		self.crop = crop
		self.grayscale = grayscale
		self.sp = sp
		self.img_root = img_root

	def read_file(self, file_path = None):
		if file_path == None:
			return []
		file_obj = open(file_path, 'rb')
		try:
			lines = file_obj.read()
		finally:
			file_obj.close()
		lines = lines.split('\n')
		data = []
		for line in lines:
			if line == '':
				continue
			data.append(line)
		return data

	
	def get_image(self, image_path):
	  	image = self.imread(image_path)
	  	return self.transform(image)


	def parse_pair(self, pair_line):
		# print pair_line
		image_path_1 = os.path.join(self.img_root,pair_line.split(self.sp)[0])
		image_path_2 = os.path.join(self.img_root,pair_line.split(self.sp)[1])
		image_label = int(pair_line.split(self.sp)[2])
		return [image_path_1, image_path_2, image_label]

	def get_image_pair(self, pair_line):
	  	[image_path_1, image_path_2, image_label] = self.parse_pair(pair_line)
	  	image_1 = self.get_image(image_path_1)
	  	image_2 = self.get_image(image_path_2)
	  	return [image_1, image_2, image_label]

	def get_inputs(self, batch_data):
  		batch = [self.get_image_pair(line) for line in batch_data]
		batch_1 =  [batch_obj[0] for batch_obj in batch] 
		batch_2 =  [batch_obj[1] for batch_obj in batch]
		batch_label =  [batch_obj[2] for batch_obj in batch]
		if self.grayscale:
			batch_inputs_1 = np.array(batch_1).astype(np.float32)[:, :, :, None]
			batch_inputs_2 = np.array(batch_2).astype(np.float32)[:, :, :, None]
		else:
			batch_inputs_1 = np.array(batch_1).astype(np.float32)
			batch_inputs_2 = np.array(batch_2).astype(np.float32)
			batch_inputs_label = np.array(batch_label).astype(np.int32)
  		return [batch_inputs_1,batch_inputs_2,batch_inputs_label]
	         
	def save_images(self, images, size, image_path):
	  	return self.imsaveBatch(inverse_transform(images), size, image_path)

	def save_image(self, image, path):
		return scipy.misc.imsave(path, image) 

	def imread(self, path):
		if (self.grayscale):
		    return scipy.misc.imread(path, flatten = True).astype(np.float)
		else:
			return scipy.misc.imread(path).astype(np.float)

	def merge_images(self, images, size):
		return self.inverse_transform(images)

	def merge(self, images, size):
		h, w = images.shape[1], images.shape[2]
		if (images.shape[3] in (3,4)):
			c = images.shape[3]
			img = np.zeros((h * size[0], w * size[1], c))
			for idx, image in enumerate(images):
			  i = idx % size[1]
			  j = idx // size[1]
			  img[j * h:j * h + h, i * w:i * w + w, :] = image
			return img
		elif images.shape[3]==1:
			img = np.zeros((h * size[0], w * size[1]))
			for idx, image in enumerate(images):
			  i = idx % size[1]
			  j = idx // size[1]
			  img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
			return img
		else:
			raise ValueError('in merge(images,size) images parameter '
		                 'must have dimensions: HxW or HxWx3 or HxWx4')

	def imsaveBatch(self, images, size, path):
		image = np.squeeze(merge(images, size))
		return scipy.misc.imsave(path, image)

	def center_crop(self, image):
		if self.crop:
			h, w = image.shape[:2]
			j = int(round((h - self.input_height)/2.))
			i = int(round((w - self.input_width)/2.))
		return scipy.misc.imresize(
			  image[j:j+self.input_height, i:i+self.input_width], \
			  [self.resize_height, self.resize_width])

	def transform(self, image):
		if self.crop:
			cropped_image = self.center_crop(image)
		else:
			cropped_image = scipy.misc.imresize(image, [self.resize_height, self.resize_width])
		return np.array(cropped_image)/127.5 - 1.

	def inverse_transform(self, images):
		return (images+1.)/2.