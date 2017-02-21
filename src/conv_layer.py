import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import pool

class conv_pool_layer(object):
    
    def __init__(self, rng, input, filter_shape, image_shape, zero_pad=True, poolsize=(2, 2), read_file=False, W_input=None, b_input=None):
        
        assert image_shape[1] == filter_shape[1]

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        if read_file==True:
        	self.W = W_input
        	self.b = b_input

        if zero_pad==True:
            input=input.transpose(2, 0, 1, 3)
            input=T.concatenate([T.shape_padleft(T.zeros_like(input[0]), 1), input, T.shape_padleft(T.zeros_like(input[0]), 1)], axis=0)
            input=input.transpose(1, 2, 0, 3)
            input=input.transpose(3, 0, 1, 2)
            input=T.concatenate([T.shape_padleft(T.zeros_like(input[0]), 1), input, T.shape_padleft(T.zeros_like(input[0]), 1)], axis=0)
            input=input.transpose(1, 2, 3, 0)
        self.input = input
        image_shape = (image_shape[0], image_shape[1], image_shape[2]+2, image_shape[3]+2)

        conv_out = conv.conv2d(
            input=self.input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            border_mode='valid'
        )

        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.switch = T.abs_(1 - T.sgn(T.abs_(conv_out - pooled_out.repeat(2, axis=2).repeat(2, axis=3))))

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]
