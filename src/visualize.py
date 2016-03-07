from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cPickle
import os

import theano
from theano import function as fn
import theano.tensor as T

from model import model

#remove
from auto_encoder import hidden_layer

load_file = open('params/trained_params.pkl', 'r')
params = ()
for i in range(12):  # update the value, depends on which layer params are being saved
	W = theano.shared(cPickle.load(load_file), borrow=True)
	b = theano.shared(cPickle.load(load_file), borrow=True)
	params+=([W, b],)
	#if (i==5):   # change for params introducing locally connected autoencoders
	#	params+=([W,b], [W,b])
load_file.close()

x = T.tensor4('x')
vis_model = model(np.random.RandomState(23455), x, (96, 336), 1, True, params)
visualize = fn([x], [vis_model.layer1.output, vis_model.layer2.output, vis_model.layer3.output, vis_model.layer4.output, vis_model.layer5.output,  vis_model.layer6.output, vis_model.layer7.output, vis_model.layer8.output, vis_model.layer9.output, vis_model.layer10.output, vis_model.layer11.output, vis_model.layer12.output ])

image = '../data/scaled-0.25/I1_000001.png'
image = Image.open(image).convert('L')
image = np.array(image, dtype='float32') / 256.
image = image.reshape(1, 1, 96, 336)
outputs = visualize(image)
'''print outputs.shape
e_inp = T.reshape(outputs, (1, 5, 504))
layer6 = hidden_layer(
	np.random.RandomState(23455),
	input = e_inp,
	n_feature_maps=5,
	n_in=504,
	n_out=40,
	b_size=1,
	read_file=True,
	W=params[5][0],
	b=params[5][1])
print layer6.output.shape.eval()
'''
image_files = []
for f in os.listdir('../data/scaled-0.25'):
	if f=='.DS_Store':
		continue
	image_files.append('../data/scaled-0.25/'+f)
image_files.sort()

rep_list = []
conf_list = []
for img in image_files:
	img = Image.open(img).convert('L')
	img = np.array(img, dtype='float32')/256.
	img = img.reshape(1, 1, 96, 336)
	representation = visualize(img)[5].flatten()
	rep_list.append(representation)

for i in range(len(rep_list)):
	temp_list = []
	for j in range(len(rep_list)):
		temp_list.append(np.sum(np.square(rep_list[i]- rep_list[j])))
	conf_list.append(temp_list)

# querying
query_index = 300
ordered_matches = list(conf_list[query_index])
ordered_matches.sort()
for i in ordered_matches[1:21]:
	print 'L2 -', i, ', frame id :', conf_list[query_index].index(i)

conf_mat = np.array(conf_list, dtype='float64')
plt.gray()
plt.subplot(1, 1, 1); 
plt.axis('off');
plt.imshow(conf_mat);
plt.show()

query_mat = np.array([conf_list[query_index]]*15, dtype='float64')
plt.gray()
plt.subplot(1, 1, 1); 
plt.axis('off');
plt.imshow(query_mat);
plt.show()

plt.gray()
plt.subplot(1, 1, 1); plt.axis('off'); plt.imshow(outputs[-1][0, 0, :, :])
plt.show()

'''
plt.subplot(10, 11, 1); plt.axis('off'); plt.imshow(image[0, 0, :, :])
layer_sizes = [2, 3, 5, 8, 5, 8, 5, 3, 2, 1]

for i in range(10):
	position = i+2
	for j in range(layer_sizes[i]):
		plt.subplot(10, 11, position); plt.axis('off'); plt.imshow(outputs[i][0, j, :, :])
		position+=11
	
plt.show()
'''
