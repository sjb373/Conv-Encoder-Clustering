from __future__ import print_function

import sys
import os
import time
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPoolLayer
import numpy as np
import theano
import theano.tensor as T
import nolearn
from nolearn.lasagne import NeuralNet
import lasagne
import matplotlib
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
from lasagne.layers import DenseLayer, InputLayer, Conv2DLayer, ReshapeLayer, DropoutLayer
def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data
        # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test
from lasagne import layers


class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(2, axis=2).repeat(2, axis=3)



X_train, y_train, X_val, y_val, X_test, y_test=load_dataset()

print( X_train.shape)
print( y_train.shape)
conv_filters=32
epochs=200
filter_sizes=3
deconv_filters=64
n_hidden=40
ae = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', ConvLayer),
        ('dropout1',layers.DropoutLayer),
        ('conv2',ConvLayer),
        ('pool2', MaxPoolLayer),        
        ('conv3',ConvLayer),
        ('dropout2',layers.DropoutLayer),
        #('conv4',ConvLayer),
        #('conv5',ConvLayer),
        ('flatten', layers.ReshapeLayer),
        #('dropout',layers.DropoutLayer),# output_dense
        ('encode_layer', layers.DenseLayer),
        ('hidden', layers.DenseLayer),  # output_dense
        ('unflatten', layers.ReshapeLayer),
        ('unpool', Unpool2DLayer),
        ('deconv', ConvLayer),
        ('output_layer', layers.ReshapeLayer),
        ('outputA',DenseLayer)
        ],
    input_shape=(None, 1, 28, 28),
    conv1_num_filters=conv_filters, conv1_filter_size = (filter_sizes, filter_sizes),
    #conv_border_mode="valid",

    conv1_nonlinearity=None,
    dropout1_p=0.15,
    conv2_num_filters=2*conv_filters, conv2_filter_size = (filter_sizes, filter_sizes),
    #conv_border_mode="valid",
    conv2_nonlinearity=None,
    pool2_pool_size=(2, 2),

    
    conv3_num_filters=3*conv_filters, conv3_filter_size = (filter_sizes, filter_sizes),
    #conv_border_mode="valid",
    conv3_nonlinearity=None,
    dropout2_p=0.25,
    #conv4_num_filters=3*conv_filters, conv4_filter_size = (filter_sizes, filter_sizes),
    #conv_border_mode="valid",
    #conv4_nonlinearity=None,
    #conv5_num_filters=8*conv_filters, conv5_filter_size = (filter_sizes, filter_sizes),
    #conv_border_mode="valid",
    #conv5_nonlinearity=None,
    flatten_shape=(([0], -1)), # not sure if necessary?
    encode_layer_num_units = n_hidden,
    hidden_num_units= deconv_filters * (28 + filter_sizes - 1) ** 2 / 4,
    unflatten_shape=(([0], deconv_filters, (28 + filter_sizes - 1) / 2, (28 + filter_sizes - 1) / 2 )),
    unpool_ds=(2, 2),
    deconv_num_filters=1, deconv_filter_size = (filter_sizes, filter_sizes),
    #deconv_border_mode="valid",
    deconv_nonlinearity=None,
    output_layer_shape = (([0], -1)),
    outputA_num_units=784,

    update_learning_rate = 0.09,
    update_momentum = 0.975,
    #batch_iterator_train=FlipBatchIterator(batch_size=128),
    regression=True,
    max_epochs= epochs,
    verbose=1,
    )
ae.fit(X_train, X_train.reshape(50000,784))

ae.save_params_to('/home/soren/Documents/ConvolutionalAutoencoders/KLP_KMEANS-master/EncoderWeights.pickle')








def autoencode_and_compare(x):
    prediction=ae.predict(x.reshape(1,1,28,28))
    f,axarr=plt.subplots(2,sharex=False)
    #axarr[0].plot()
    #axD=fig.add_subplot(1,1,1)
    #axE=fig.add_subplot(1,1,1)
    axarr[0].matshow(x.reshape(28,28),cmap=matplotlib.cm.binary)
    axarr[1].matshow(prediction.reshape(28,28),cmap=matplotlib.cm.binary)
    plt.show()
### check it worked :


def plot_random(X):
    idx=np.random.randint(low=0,high=X.shape[0])
    autoencode_and_compare(X_train[idx,:,:,:])


EncodedOutputNet=NeuralNet(
    layers=[
        ('input', layers.InputLayer),

        ('conv1', ConvLayer),
        ('dropout1',layers.DropoutLayer),
        ('conv2',ConvLayer),
        ('pool2', MaxPoolLayer),        
        ('conv3',ConvLayer),
        ('dropout2',layers.DropoutLayer),
        #('conv4',ConvLayer),
        #('conv5',ConvLayer),
        ('flatten', layers.ReshapeLayer),
        #('dropout',layers.DropoutLayer),# output_dense
        ('encode_layer', layers.DenseLayer),
        
        ],
    input_shape=(None, 1, 28, 28),
    conv1_num_filters=conv_filters, conv1_filter_size = (filter_sizes, filter_sizes),
    #conv_border_mode="valid",

    conv1_nonlinearity=None,
    dropout1_p=0.15,
    conv2_num_filters=2*conv_filters, conv2_filter_size = (filter_sizes, filter_sizes),
    #conv_border_mode="valid",
    conv2_nonlinearity=None,
    pool2_pool_size=(2, 2),

    
    conv3_num_filters=3*conv_filters, conv3_filter_size = (filter_sizes, filter_sizes),
    #conv_border_mode="valid",
    conv3_nonlinearity=None,
    dropout2_p=0.25,
    #conv4_num_filters=3*conv_filters, conv4_filter_size = (filter_sizes, filter_sizes),
    #conv_border_mode="valid",
    #conv4_nonlinearity=None,
    #conv5_num_filters=8*conv_filters, conv5_filter_size = (filter_sizes, filter_sizes),
    #conv_border_mode="valid",
    #conv5_nonlinearity=None,
    flatten_shape=(([0], -1)), # not sure if necessary?
    encode_layer_num_units = n_hidden,
    

    update_learning_rate = 0.09,
    update_momentum = 0.975,
    #batch_iterator_train=FlipBatchIterator(batch_size=128),
    regression=True,
    max_epochs= epochs,
    verbose=1,
    )
#load the learned params
EncodedOutputNet.load_params_from('/home/soren/Documents/ConvolutionalAutoencoders/KLP_KMEANS-master/EncoderWeights.pickle')

#get
EncodedOutputs=EncodedOutputNet.predict(X_test)
print('Shape of matrix of encoded features:{}'.format(EncodedOutputs.shape))

kmeans_instance = KMeans(n_clusters=n_digits,)
kmeans_instance.fit(EncodedOutputs)

def plot_cluster_sample(Kmeans,Enc,X,i,s=10):
    """Takes a kmeans cluster instance, mnist data x
    and plots random mnist pictures in the cluster,
    so i can decide which digit the cluster represents best
     """
    X=X.reshape(X.shape[0],1,28,28)
    Kmeans.cluster_centers_[i]
    samples=[]
    while len(samples) <= s**2:
      idx=np.random.randint(low=0,high=X.shape[0])
      cluster=Kmeans.predict(Enc[idx,:].reshape(n_hidden))
      #print('data idx: {} | Cluster Index: {}'.format(idx,cluster))
      if cluster[0] == i and idx not in samples:
        samples.append(idx)

    #f,axarr=plt.subplot(samples_needed,sharex=False)
    fig = plt.figure()
    plt.title('Random Sample of digits in cluster {}'.format(i))
    j=0
    for x in range(s):
      for y in range(s):
        image=X[samples[j],:,:,:].reshape(28,28)

        #print("image shape:{}".format(image.shape))
        ax=fig.add_subplot(s,s,s*y+x)
        ax.matshow(image, cmap = matplotlib.cm.binary)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        j+=1
    plt.show()

plot_cluster_sample(kmeans_instance,EncodedOutputs,X_test,0,s=10)