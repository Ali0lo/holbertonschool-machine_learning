#!/usr/bin/env python3
"""Module for creating a sparse autoencoder."""
import tensorflow.keras as keras
def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Create a sparse autoencoder with L1 regularization on the latent layer.
    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in the
            encoder. The hidden layers are reversed for the decoder.
        latent_dims (int): Dimensions of the latent space representation.
        lambtha (float): L1 regularization parameter applied to the encoded
            output.
    Returns:
        tuple: (encoder, decoder, auto)
            - encoder: the encoder model
            - decoder: the decoder model
            - auto: the sparse autoencoder model
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.L1(lambtha)
    )(x)
    encoder = keras.Model(encoder_input, latent)
    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, decoder_output)
    # Autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
