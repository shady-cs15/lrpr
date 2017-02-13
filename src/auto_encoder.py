import numpy as np
# Hello Shady

import theano
import theano.tensor as T

class hidden_layer(object):
	def __init__(self, rng, input, n_feature_maps, n_in, n_out, b_size=5, read_file=False, W=None, b=None):
		
		# input dim should be: batch_size x n_feature_maps x 504
		# n_in and n_out should be 504 and 40 respectively
		input = T.transpose(input, (1, 0, 2))
		self.input = input
		if read_file==False:
			W_values = np.asarray(
				rng.uniform(
					low=-np.sqrt(6./(n_in+n_out)),
					high=np.sqrt(6./(n_in+n_out)),
					size=(n_in, n_out)
				),
				dtype=theano.config.floatX
			)
			
			W = theano.shared(value=W_values, name='W', borrow=True)

			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b

		embedding_list = []
		for i in range(n_feature_maps):
			embedding_list.append(T.tanh(T.dot(input[i], self.W) + self.b))
		self.output = T.concatenate(embedding_list, axis=0)
		self.output = T.reshape(self.output, (n_feature_maps, b_size, n_out))
		self.params = [self.W, self.b]

		self.input = T.transpose(self.input, (1, 0, 2))
		self.output = T.transpose(self.output, (1, 0, 2))
