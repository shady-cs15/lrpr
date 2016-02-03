from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cPickle

import os
import numpy
import sys

import theano
from theano import function as fn #remove
import theano.tensor as T #remove

from model import model

def usage():
	print 'Usage: python train_model.py img_h img_w'
	sys.exit(1)

if len(sys.argv) < 3:
	usage()

img_h = int(sys.argv[1])
img_w = int(sys.argv[2])


''' read image files from dir
	scaled-0.25 dir contains Karlsruhe sequence
	_0081 scaled by a factor of 0.25
'''
image_files = []
for f in os.listdir('../data/scaled-0.25'):
	if f=='.DS_Store':
		continue
	image_files.append('../data/scaled-0.25/'+f)
image_files.sort()



''' load data into train set & validation set
	test set to be loaded and tested separately
'''
# TODO - valid set
train_set = ()
valid_set = ()
test_set = ()
for f in image_files[:1]:   #TODO- remove restriction
	img = Image.open(f).convert('L')
	img = np.array(img, dtype='float64') / 256.
	img = img.reshape(1, 1, img_h, img_w)
	train_set+=(img, )
train_set = numpy.concatenate(train_set, axis=0)
train_set = theano.shared(np.asarray(train_set, dtype=theano.config.floatX), borrow=True)


''' function to train the model
	data_set : (train_set, valid_set, test_set)
	batch_size : mini batch size for SGD
	learning_rate : initial learning rate
	init : if True, parameters are passed as params argument
	params: if init is true, pass parameters
'''
# TODO - make adaptive learning rate

rng = np.random.RandomState(23455)
dummy_wt = theano.shared(numpy.asarray(rng.uniform(low=-1., high=-1., size=(1, 1)), dtype=theano.config.floatX), borrow=True)
dummy_wt = ([dummy_wt, ]*2, )*12

def train_nnet(rng, data_set, batch_size=5, learning_rate=0.07, init=False, params=dummy_wt):
	train_model = model(rng, data_set[0], (img_h, img_w), batch_size=batch_size, params=params)

	# remove the following block made for visualisation
	# block starts here
	inp = T.tensor4()
	train_ = model(rng, inp, (img_h, img_w), batch_size=batch_size, params=params)
	f = fn([inp], train_.layer12.output)
	f_img = f(data_set[0].eval())
	plt.gray()
	for i in range(1, 2):
		plt.subplot(1, 1, i); plt.axis('off'); plt.imshow(f_img[0, i-1, :, :])
	plt.show()
	# block ends here

train_nnet(rng, (train_set, ), 1)
