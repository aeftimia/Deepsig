import keras
import numpy
import matplotlib.pyplot as plt
import os
import random
import tensorflow

from keras import backend as K
from keras.datasets import mnist
from keras.engine.topology import Layer
from keras.layers import Activation, Dense, Input, Lambda
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.losses import binary_crossentropy
from keras.models import Model
from PIL import Image
from sklearn.mixture import GaussianMixture
from scipy.stats import mode

numpy.random.seed(42)

# Network parameters
batch_size = 128
num_epochs = 30
kernel_size = 4
latent_dims = [16, 2]
strides=2
layer_filters = [16, 32]

# mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

def preprocess(data):
    if data.ndim == 3:
        data = numpy.asarray([data]).transpose((1, 2, 3, 0))
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
layer.activation = Activation(None)

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
encoder_mean = Model(common_input, z_mean, name='encoder_mean')
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
    sample_mean = K.mean(z_mean, 0)
    # large batch size ~> unbiased estimator
    sample_log_var = K.log(K.mean(K.exp(z_log_var), 0))
    kl_loss = K.sum((-z_log_var + K.square(z_mean) + K.exp(z_log_var)) / 2, axis=-1)
    reconstruction_loss = binary_crossentropy(K.flatten(yTrue), K.flatten(yPred)) * numpy.prod(x_train.shape[1:])
    return K.mean(reconstruction_loss + kl_loss)

autoencoder.compile(loss=elbo_loss, optimizer='adam')

# Train the autoencoder
autoencoder.fit(x_train,
        x_train,
        validation_data=(x_test, x_test),
        epochs=num_epochs,
        batch_size=batch_size)

# Train reconstruction with discriminator
def generate_adversarial_data(sample_data):
    x_encoded = encoder.predict(sample_data)[1]
    fake_encode = numpy.random.normal(size=(len(sample_data), latent_dims[-1]))
    x = numpy.concatenate((x_encoded, fake_encode))
    y = numpy.concatenate((numpy.ones(len(x_encoded)), numpy.zeros((len(fake_encode)))))
    indices = numpy.arange(len(x))
    numpy.random.shuffle(indices)
    x = x[indices]
    return x, y[indices]

x = Dense(64, activation='relu')(decoder_input)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(1, activation='sigmoid', name='discriminator')(x)

for _ in range(10):
    gan_train_x, gan_train_y = generate_adversarial_data(x_train)
    gan_test_x, gan_test_y = generate_adversarial_data(x_test)
    discriminator = Model(decoder_input, x, name='discriminator')
    discriminator.compile(loss='binary_crossentropy', optimizer='adam')
    discriminator.fit(gan_train_x,
            gan_train_y,
            validation_data=(gan_test_x, gan_test_y),
            epochs=num_epochs,
            batch_size=batch_size)

    gan_train_x, gan_train_y = generate_adversarial_data(x_train)
    discriminator = Model(decoder_input, x, name='discriminator')
    discriminator.compile(loss='binary_crossentropy', optimizer='adam')
    discriminator.trainable = False
    decoder_trainer = Model(decoder_input, discriminator(encoder_mean(decoder(decoder_input))), name='autoencoder_trainer')
    decoder_trainer.compile(loss='binary_crossentropy', optimizer='adam')
    decoder_trainer.fit(gan_train_x,
            numpy.ones(len(gan_train_x)),
            validation_data=(gan_test_x, numpy.ones(len(gan_test_x))),
            epochs=num_epochs,
            batch_size=batch_size)

n_clusters = 10
clusterer = GaussianMixture(n_components=n_clusters, covariance_type='diag', max_iter=1000)
clusterer.fit(encoder.predict(x_train)[1])
predictions = clusterer.predict(encoder.predict(x_test)[1])

scores = []
y_test = y_test.flatten()
print(y_test[:10])
print('pred')
print(predictions[:10])
for i in range(n_clusters):
    k = mode(predictions[y_test == i]).mode[0]
    scores.append(numpy.logical_and(y_test == i, predictions == k).sum() / sum(predictions == k) * 100)

print(numpy.mean(scores))
print('\n'.join(map(str, scores)))
y_random = y_test.copy()
numpy.random.shuffle(y_random)
print(y_random[:10])
print('pred')
print(predictions[:10])
scores = []
for i in range(n_clusters):
    k = mode(predictions[y_random == i]).mode[0]
    scores.append(numpy.logical_and(y_random == i, predictions == k).sum() / sum(predictions == k) * 100)

print(numpy.mean(scores))
print('\n'.join(map(str, scores)))

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    _, z_mean, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent_gan.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = x_test.shape[1]
    figure = numpy.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = numpy.linspace(-4, 4, n)
    grid_y = numpy.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = numpy.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(x_train.shape[1:])
            if digit.ndim == 3:
                digit = digit.mean(2)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = numpy.arange(start_range, end_range, digit_size)
    sample_range_x = numpy.round(grid_x, 1)
    sample_range_y = numpy.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

plot_results((encoder, decoder), (x_test, y_test), batch_size=batch_size, model_name="vae_cnn")
