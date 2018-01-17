# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, LSTM, GRU, RepeatVector
from keras.layers.core import Dense, Lambda
from keras import objectives
import numpy as np
import matplotlib.pyplot as plt


def create_lsm_vae(input_dim,
                   timesteps,
                   batch_size,
                   intermediate_dim,
                   latent_dim,
                   epsilon_std=1.,
                   model_option='lstm'):
    x = Input(shape=(timesteps, input_dim,))

    if model_option == 'lstm':
        h = LSTM(intermediate_dim)(x)
    elif model_option == 'gru':
        h = GRU(intermediate_dim)(x)

    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0, stddev=epsilon_std)
        return z_mean + z_log_sigma + epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    if model_option == 'lstm':
        decoder_h = LSTM(intermediate_dim, return_sequences=True)
        decoder_mean = LSTM(input_dim, return_sequences=True)
    elif model_option == 'gru':
        decoder_h = GRU(intermediate_dim, return_sequences=True)
        decoder_mean = GRU(input_dim, return_sequences=True)

    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    x_decoded_mean = decoder_mean(h_decoded)

    vae = Model(x, x_decoded_mean)
    vae.summary()

    encoder = Model(x, z_mean)

    decoder_input = Input(shape=(latent_dim,))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)

    _x_decode_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decode_mean)

    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    vae.compile(optimizer='adam', loss=vae_loss)

    return vae, encoder, generator


def get_data():
    data = np.fromfile("data/sample_data.dat").reshape(419, 13)
    timestep = 3
    dataX = []
    for i in range(len(data) - timestep - 1):
        x = data[i:(i + timestep), :]
        dataX.append(x)
    return np.array(dataX)


x = get_data()
input_dim = x.shape[-1]
timesteps = x.shape[1]
batch_size = 1

vae, enc, gen = create_lsm_vae(input_dim,
                               timesteps=timesteps,
                               batch_size=batch_size,
                               intermediate_dim=16,
                               latent_dim=25,
                               epsilon_std=0.25,
                               model_option='lstm')

vae.fit(x, x, epochs=100)

preds = vae.predict(x, batch_size=batch_size)

plt.plot(x[:, 0, 3], label='data')
plt.plot(preds[:, 0, 3], label='predict')
plt.legend()
plt.show()
