from conv_layer import conv_pool_layer
from deconv_layer import deconv_unpool_layer

import theano
import numpy as np

class model(object):

	def __init__(self, rng, input, input_dim, batch_size=5, init=False, params=None):
		
		self.input = input
		self.inp_h = input_dim[0]
		self.inp_w = input_dim[1]
		
		self.layer1 = conv_pool_layer(
			rng,
			input = input,
			image_shape=(batch_size, 1, self.inp_h, self.inp_w),
			filter_shape=(2, 1, 3, 3),
			poolsize=(1, 1),
			zero_pad=True,
			read_file=init,
			W_input=params[0][0],
			b_input=params[0][1]
		)

		self.layer2 = conv_pool_layer(
			rng,
			input = self.layer1.output,
			image_shape=(batch_size, 2, self.inp_h/1, self.inp_w/1),
			filter_shape=(3, 2, 3, 3),
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[1][0],
			b_input=params[1][1]
		)

		self.layer3 = conv_pool_layer(
			rng,
			input = self.layer2.output,
			image_shape=(batch_size, 3, (self.inp_h/1)/2, (self.inp_w/1)/2),
			filter_shape=(5, 3, 3, 3),
			poolsize=(1, 1),
			zero_pad=True,
			read_file=init,
			W_input=params[2][0],
			b_input=params[2][1]
		)

		self.layer4 = conv_pool_layer(
			rng,
			input = self.layer3.output,
			image_shape=(batch_size, 5, (self.inp_h/2), (self.inp_w)/2),
			filter_shape=(8, 5, 3, 3),
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[3][0],
			b_input=params[3][1]
		)

		self.layer5 = conv_pool_layer(
			rng,
			input = self.layer4.output,
			image_shape=(batch_size, 8, (self.inp_h/2)/2, (self.inp_w/2)/2),
			filter_shape=(10, 8, 3, 3),
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[4][0],
			b_input=params[4][1]
		)

		self.layer8 = deconv_unpool_layer(
			rng,
			input = self.layer5.output,
			image_shape=(batch_size, 10, ((self.inp_h/2)/2)/2, ((self.inp_w/2)/2)/2),
			filter_shape=(8, 10, 3, 3),
			unpoolsize=(2, 2),
			zero_pad=True,
			switch=self.layer5.switch,
			read_file=init,
			W_input=params[7][0],
			b_input=params[7][1]
		)

		self.layer9 = deconv_unpool_layer(
			rng,
			input = self.layer8.output,
			image_shape=(batch_size, 8, (self.inp_h/2)/2, (self.inp_w/2)/2),
			filter_shape=(5, 8, 3, 3),
			unpoolsize=(2, 2),
			zero_pad=True,
			switch=self.layer4.switch,
			read_file=init,
			W_input=params[8][0],
			b_input=params[8][1]
		)

		self.layer10 = deconv_unpool_layer(
			rng,
			input=self.layer9.output,
			image_shape=(batch_size, 5, self.inp_h/2, self.inp_w/2),
			filter_shape=(3, 5, 3, 3),
			unpoolsize=(1, 1),
			switch=None,
			read_file=init,
			W_input=params[9][0],
			b_input=params[9][1]
		)

		self.layer11 = deconv_unpool_layer(
			rng,
			input=self.layer10.output,
			image_shape=(batch_size, 3, self.inp_h/2, self.inp_w/2),
			filter_shape=(2, 3, 3, 3),
			unpoolsize=(2, 2),
			switch=self.layer2.switch,
			read_file=init,
			W_input=params[10][0],
			b_input=params[10][1]
		)

		self.layer12 = deconv_unpool_layer(
			rng,
			input=self.layer11.output,
			image_shape=(batch_size, 2, self.inp_h, self.inp_w),
			filter_shape=(1, 2, 3, 3),
			unpoolsize=(1, 1),
			switch=None,
			read_file=init,
			W_input=params[11][0],
			b_input=params[11][1]
		)