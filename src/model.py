from conv_layer import conv_pool_layer

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
			poolsize=(2, 2),
			zero_pad=True,
			read_file=init,
			W_input=params[0][0],
			b_input=params[0][1]
		)