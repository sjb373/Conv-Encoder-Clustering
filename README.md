# Conv-Encoder-Clustering
Trains a convolutional autoencoder on MNIST, then runs k-means on the encoded images to group them into 10 clusters. 

Just an idea I thought was interesting and easy to implement. Doesn't work very well, it actually performs slightly worse than just running k-means on raw mnist pixels.


Dependencies: lasagne+theano+nolearn+scikit-learn.
