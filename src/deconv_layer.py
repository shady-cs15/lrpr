import theano.tensor as T
import theano
from theano.tensor.nnet import conv

import numpy

class deconv_unpool_layer(object):

	def unpool(self, input, ds, switch=None, pad_bottom=False, pad_right=False):
		output = input.repeat(ds[0], axis=2).repeat(ds[1], axis=3)

		if pad_bottom==True:
			output = output.transpose(2, 0, 1, 3)
			output = T.concatenate([output, T.shape_padleft(output[-1], 1)], axis=0)
			output = output.transpose(1, 2, 0, 3)
		if pad_right==True:
			output = output.transpose(3, 0, 1, 2)
			output = T.concatenate([output, T.shape_padleft(output[-1], 1)], axis=0)
			output = output.transpose(1, 2, 3, 0)

		if switch!=None:
			output=output*switch
		
		return output


	def __init__(self, rng, input, filter_shape, image_shape, unpoolsize=(2, 2), switch=None, zero_pad=True, pad_bottom=False, pad_right=False, read_file=False, W_input=None, b_input=None, non_linearity=False):
	
		assert image_shape[1] == filter_shape[1]
		self.input = input

		unpooled_out = self.unpool(
			input = input,
			ds = unpoolsize,
			switch = switch,
			pad_bottom = pad_bottom,
			pad_right = pad_right
		)

		if zero_pad==True:
			input=unpooled_out.transpose(2, 0, 1, 3)
			input=T.concatenate([T.shape_padleft(T.zeros_like(input[0]), 1), input, T.shape_padleft(T.zeros_like(input[0]), 1)], axis=0)
			input=input.transpose(1, 2, 0, 3)
			input=input.transpose(3, 0, 1, 2)
			input=T.concatenate([T.shape_padleft(T.zeros_like(input[0]), 1), input, T.shape_padleft(T.zeros_like(input[0]), 1)], axis=0)
			input=input.transpose(1, 2, 3, 0)


		fan_in = numpy.prod(filter_shape[1:])
		fan_out = (filter_shape[0]*numpy.prod(filter_shape[2:])/numpy.prod(unpoolsize))
		W_bound = numpy.sqrt(6./(fan_in+fan_out))
		self.W = theano.shared(
			numpy.asarray(
				rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
				dtype=theano.config.floatX
			),
			borrow=True
		)
		b_values = numpy.zeros((filter_shape[0], ), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, borrow=True)

		if read_file==True:
			self.W = W_input
			self.b = b_input

		image_ht = image_shape[2]*unpoolsize[0] + 2
		image_wd = image_shape[3]*unpoolsize[1] + 2
		if pad_bottom==True:
			image_ht+=1
		if pad_right==True:
			image_wd+=1
		image_shape = (image_shape[0], image_shape[1], image_ht, image_wd)

		deconv_out = conv.conv2d(
			input=input,
			filters = self.W,
			filter_shape=filter_shape,
			image_shape=image_shape,
			border_mode='valid'
		)

		self.output = (deconv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		if non_linearity==True:
			self.output = T.tanh(self.output)

		self.params = [self.W, self.b]