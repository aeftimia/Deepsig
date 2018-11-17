import keras
import numpy
import matplotlib.pyplot as plt
import random
import tensorflow

from keras import backend as K
from keras.datasets import mnist
from keras.engine.topology import Layer
from keras.layers import Activation, Dense, Input, Lambda
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose, BatchNormalization
from keras.losses import binary_crossentropy
from keras.models import Model
from PIL import Image
from sklearn.mixture import GaussianMixture
from scipy.stats import mode

numpy.random.seed(42)

# Network parameters
batch_size = 128
num_epochs = 10
kernel_size = 3
latent_dims = [128, 3]
strides=2
layer_filters = [32, 64]

# mnist dataset
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
for latent_dim in latent_dims[:-1]:
    layer = Dense(latent_dim, activation='relu')
    x = layer(x)

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

latent_dim = latent_dims[-1]
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
# z = Dense(latent_dim, name='latent_layer')(x)
# layer.activation=Activation(None)

encoder = Model(common_input, [z, z_mean, z_log_var], name='encoder')
encoder.summary()

# Decoder
decoder_input = Input(shape=(latent_dim,), name='decoder_input')
x = decoder_input
for i, latent_dim in enumerate(latent_dims[-2::-1] + [numpy.prod(encoder_layers[-1].output_shape[1:])]):
    layer = Dense(latent_dim, activation='relu')
    if i == 0:
        layer.activation = Activation(None)
    x = layer(x)
x = Reshape(conv_shape)(x)

for layer in decoder_layers[::-1]:
    x = layer(x)
decoder = Model(decoder_input, x, name='decoder')
decoder.summary()

# Autoencoder
autoencoder_output = decoder(encoder(common_input)[0])
autoencoder = Model(common_input, autoencoder_output, name='autoencoder')
autoencoder.summary()

def elbo_loss(yTrue, yPred):
    kl_loss = K.sum(z_log_var - (K.square(z_mean) + K.exp(z_log_var)) / 2, axis=-1)
    reconstruction_loss = binary_crossentropy(K.flatten(yTrue), K.flatten(yPred)) * numpy.prod(x_train.shape[1:])
    return K.mean(reconstruction_loss - kl_loss)

autoencoder.compile(loss=elbo_loss, optimizer='adam')

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
clusterer = GaussianMixture(n_components=n_clusters, covariance_type='diag', max_iter=1000)
clusterer.fit(encoder.predict(x_train)[1])
predictions = clusterer.predict(encoder.predict(x_test)[1])

scores = []
y_test = y_test.flatten()
for i in range(n_clusters):
    k = mode(predictions[y_test == i])
    scores.append(numpy.logical_and(y_test == i, predictions == k).sum() / sum(y_test == i) * 100)

print(numpy.mean(scores))
print('\n'.join(map(str, scores)))
y_random = y_test.copy()
numpy.random.shuffle(y_random)
scores = []
for i in range(n_clusters):
    k = mode(predictions[y_random == i])
    scores.append(numpy.logical_and(y_random == i, predictions == k).sum() / sum(y_random == i) * 100)

print('\n'.join(map(str, scores)))
print(numpy.mean(scores))
