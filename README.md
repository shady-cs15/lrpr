# Learning representations for place recognition <br>
Uses convolutional autoencoders (with certain modfications) to extract features from raw images. The idea of learning such dense representations is to map the images into dense vectors and compare them in the low-dimensional vector space. <br>
<img src="https://raw.githubusercontent.com/shady-cs15/shady-cs15.github.io/master/images/lrpr1.png"/> <br>

The architecture is a 12 layer convolutional autoencoder with 6 encoding layers and 6 decoding layers. The model is first pretrained in a greedy layer wise fashion and then fine tuned with respect to a global criterion. The following figure shows the architecture.
<br>
<img src="https://raw.githubusercontent.com/shady-cs15/shady-cs15.github.io/master/images/lrpr2.png"/> <br>

report at <a href="https://github.com/shady-cs15/shady-cs15.github.io/blob/master/files/lrpr.pdf"/>this link</a>

# Dependencies (Python)
1. Theano <br>
2. Matplotlib <br>
3. cPickle <br>
4. numpy <br>
5. pillow <br>

# Running
```
python train_model.py
```
<br>Running on smaller data <br>
```
python visualise.py
``` 

# Running with GPU
Make sure proper driver and Nvidia CUDA toolkit is installed.
add the following lines to ~/.theanorc
```
[global]
device=gpu
floatX=float32
```
