#!/usr/bin/env python3
"""Module for creating a convolutional autoencoder."""
import tensorflow.keras as keras
def autoencoder(input_dims, filters, latent_dims):
    """Create a convolutional autoencoder.
    Args:
        input_dims (tuple): Dimensions of the model input (H, W, C).
        filters (list): Number of filters for each convolutional layer in
            the encoder. Reversed for the decoder.
        latent_dims (tuple): Dimensions of the latent space (H, W, C).
    Returns:
        tuple: (encoder, decoder, auto)
            - encoder: the encoder model
            - decoder: the decoder model
            - auto: the full autoencoder model
    """
    # Encoder
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(encoder_input, x)
    # Decoder
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input
    reversed_filters = list(reversed(filters))
    # All layers except the last: same padding + upsample
    for f in reversed_filters[:-1]:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    # Second to last convolution: valid padding + upsample
    x = keras.layers.Conv2D(reversed_filters[-1], (3, 3), activation='relu',
                            padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    # Last convolution: channels filters, sigmoid, same padding, no upsample
    channel = input_dims[-1]
    decoder_output = keras.layers.Conv2D(channel, (3, 3), activation='sigmoid',
                                         padding='same')(x)
    decoder = keras.Model(decoder_input, decoder_output)
    # Autoencoder
    auto_input = keras.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
