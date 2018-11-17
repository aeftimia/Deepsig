import keras
import numpy
import matplotlib.pyplot as plt
import random
import tensorflow

from keras import backend as K
from keras.datasets import mnist
from keras.engine.topology import Layer
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose, BatchNormalization
from keras.models import Model
from PIL import Image
from sklearn.cluster import KMeans
from scipy.stats import mode

numpy.random.seed(42)

# Network parameters
batch_size = 128
num_epochs = 10
kernel_size = 3
latent_dims = [128, 2]
strides=2
layer_filters = [32, 64]

# CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def preprocess(data):
    if data.ndim == 3:
        data = numpy.asarray([data]).transpose((1, 2, 3, 0))
    print(data.shape)
    maxima = data.max(axis=tuple(range(1, data.ndim))).reshape((len(data),) + (1,) * (data.ndim - 1))
    return data.astype('float32') / maxima, maxima

image_size = x_train.shape[1]
x_train, train_decode = preprocess(x_train)
input_shape = x_train.shape[1:]
num_channels = x_train.shape[-1]
x_test, test_decode = preprocess(x_test)

x_train = numpy.clip(x_train, 0., 1.)
x_test = numpy.clip(x_test, 0., 1.)

encoder_layers = []
for filters in layer_filters:
    encoder_layers.append(Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               activation='relu',
               padding='same'))

decoder_layers = []
for filters in [num_channels] + layer_filters[:-1]:
    decoder_layers.append(Conv2DTranspose(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               activation='relu',
               padding='same'))
decoder_layers[0].activation = Activation('sigmoid')

# Encoder
common_input = Input(shape=input_shape, name='encoder_input')
x = common_input
for layer in encoder_layers:
    x = layer(x)

conv_shape = K.int_shape(x)[1:]
x = Flatten()(x)
flat_shape = K.int_shape(x)[1:]

# Latent Layer
for latent_dim in latent_dims:
    layer = Dense(latent_dim, activation='relu')
    x = layer(x)
layer.activation=Activation(None)
layer.name='latent_layer'

encoder = Model(common_input, x, name='encoder')
encoder.summary()

# Decoder
decoder_input = Input(shape=(latent_dim,), name='decoder_input')
x = decoder_input
for i, latent_dim in enumerate(latent_dims[::-1]):
    layer = Dense(flat_shape[0], activation='relu')
    if i == 0:
        layer.activation = Activation(None)
    x = layer(x)
x = Reshape(conv_shape)(x)

for layer in decoder_layers[::-1]:
    x = layer(x)
decoder = Model(decoder_input, x, name='decoder')
decoder.summary()

# Autoencoder
autoencoder = Model(common_input, decoder(encoder(common_input)), name='autoencoder')
autoencoder.summary()

autoencoder.compile(loss='mse', optimizer='adam')

# Train the autoencoder
autoencoder.fit(x_train,
        x_train,
        validation_data=(x_test, x_test),
        epochs=num_epochs,
        batch_size=batch_size)

# Test reconstruction
x_decoded = autoencoder.predict(x_test)

# Display the 1st 8 reconstructed images
rows, cols = 10, 30
num = rows * cols
imgs = numpy.concatenate([(x_test * test_decode)[:num], (x_decoded * test_decode)[:num]])
imgs = imgs.reshape((rows * 2, cols) + x_test.shape[1:])
imgs = numpy.vstack(numpy.split(imgs, 2, axis=1))
imgs = imgs.reshape((rows * 2, -1,) + x_test.shape[1:])
imgs = numpy.vstack([numpy.hstack(i) for i in imgs])
imgs = imgs.astype(numpy.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
# plt.imshow(imgs, interpolation='none', cmap='gray')
# Image.fromarray(imgs).save('corrupted_and_denoised.png')

n_clusters = 10
clusterer = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, max_iter=300, n_jobs=-1)
clusterer.fit(encoder.predict(x_train))
predictions = clusterer.predict(encoder.predict(x_test))

scores = []
y_test = y_test.flatten()
for i in range(n_clusters):
    k = mode(predictions[y_test == i])
    scores.append(numpy.logical_and(y_test == i, predictions == k).sum() / sum(y_test == i) * 100)

y_random = y_test.copy()
numpy.random.shuffle(y_random)
for i in range(n_clusters):
    k = mode(predictions[y_random == i])
    scores.append(numpy.logical_and(y_random == i, predictions == k).sum() / sum(y_random == i) * 100)

print('\n'.join(map(str, scores)))
