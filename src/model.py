from conv_layer import conv_pool_layer
from deconv_layer import deconv_unpool_layer
from auto_encoder import hidden_layer

import theano
import theano.tensor as T

import numpy as np

class model(object):

	def __init__(self, rng, input, input_dim, batch_size=5, init=False, params=None):
		
		self.input = input
		self.inp_h = input_dim[0]
		self.inp_w = input_dim[1]
		
		self.layer1 = conv_pool_layer(
			rng,
			input = input,
			image_shape=(batch_size, 3, self.inp_h, self.inp_w),
			filter_shape=(16, 3, 3, 3),
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[0][0],
			b_input=params[0][1]
		)

		self.layer2 = conv_pool_layer(
			rng,
			input = self.layer1.output,
			image_shape=(batch_size, 16, self.inp_h/2, self.inp_w/2),
			filter_shape=(32, 16, 3, 3),
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[1][0],
			b_input=params[1][1]
		)

		self.layer3 = conv_pool_layer(
			rng,
			input = self.layer2.output,
			image_shape=(batch_size, 32, self.inp_h/4, self.inp_w/4),
			filter_shape=(64, 32, 3, 3),
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[2][0],
			b_input=params[2][1]
		)

		self.layer4 = conv_pool_layer(
			rng,
			input = self.layer3.output,
			image_shape=(batch_size, 64, self.inp_h/8, self.inp_w/8),
			filter_shape=(128, 64, 3, 3),
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[3][0],
			b_input=params[3][1]
		)

		self.layer5 = conv_pool_layer(
			rng,
			input = self.layer4.output,
			image_shape=(batch_size, 128, self.inp_h/16, self.inp_w/16),
			filter_shape=(256, 128, 3, 3),
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[4][0],
			b_input=params[4][1]
		)

		self.layer6 = conv_pool_layer(
			rng,
			input = self.layer5.output,
			image_shape=(batch_size, 256, self.inp_h/32, self.inp_w/32),
			filter_shape=(512, 256, 3, 3),
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[5][0],
			b_input=params[5][1]
		)

		self.layer7 = deconv_unpool_layer(
			rng,
			input = self.layer6.output,
			image_shape=(batch_size, 512, self.inp_h/64, self.inp_w/64),
			filter_shape=(256, 512, 3, 3),
			unpoolsize=(2, 2),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer6.switch,
			read_file=init,
			W_input=params[6][0],
			b_input=params[6][1]
		)

		self.layer8 = deconv_unpool_layer(
			rng,
			input = self.layer7.output,
			image_shape=(batch_size, 256, self.inp_h/32, self.inp_w/32),
			filter_shape=(128, 256, 3, 3),
			unpoolsize=(2, 2),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer5.switch,
			read_file=init,
			W_input=params[7][0],
			b_input=params[7][1]
		)

		self.layer9 = deconv_unpool_layer(
			rng,
			input = self.layer8.output,
			image_shape=(batch_size, 128, self.inp_h/16, self.inp_w/16),
			filter_shape=(64, 128, 3, 3),
			unpoolsize=(2, 2),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer4.switch,
			read_file=init,
			W_input=params[8][0],
			b_input=params[8][1]
		)

		self.layer10 = deconv_unpool_layer(
			rng,
			input = self.layer9.output,
			image_shape=(batch_size, 64, self.inp_h/8, self.inp_w/8),
			filter_shape=(32, 64, 3, 3),
			unpoolsize=(2, 2),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer3.switch,
			read_file=init,
			W_input=params[9][0],
			b_input=params[9][1]
		)

		self.layer11 = deconv_unpool_layer(
			rng,
			input = self.layer10.output,
			image_shape=(batch_size, 32, self.inp_h/4, self.inp_w/4),
			filter_shape=(16, 32, 3, 3),
			unpoolsize=(2, 2),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer2.switch,
			read_file=init,
			W_input=params[10][0],
			b_input=params[10][1]
		)

		self.layer12 = deconv_unpool_layer(
			rng,
			input = self.layer11.output,
			image_shape=(batch_size, 16, self.inp_h/2, self.inp_w/2),
			filter_shape=(3, 16, 3, 3),
			unpoolsize=(2, 2),
			zero_pad=True,
			non_linearity=True,
			switch=self.layer1.switch,
			read_file=init,
			W_input=params[11][0],
			b_input=params[11][1]
		)
