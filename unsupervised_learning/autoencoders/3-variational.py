#!/usr/bin/env python3
"""Module for creating a variational autoencoder."""
import tensorflow.keras as keras
def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder.
    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in the
            encoder. The hidden layers are reversed for the decoder.
        latent_dims (int): Dimensions of the latent space representation.
    Returns:
        tuple: (encoder, decoder, auto)
            - encoder: outputs (z, z_mean, z_log_var)
            - decoder: the decoder model
            - auto: the full autoencoder model compiled with adam and
              binary cross-entropy + KL divergence loss
    """
    import tensorflow as tf
    # --- Encoder ---
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)
    def sampling(args):
        """Sample z via reparameterization trick."""
        mean, log_var = args
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(log_var / 2) * eps
    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = keras.Model(encoder_input, [z, z_mean, z_log_var])
    # --- Decoder ---
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, decoder_output)
    # --- Autoencoder ---
    auto_input = keras.Input(shape=(input_dims,))
    z_out, z_mean_out, z_log_var_out = encoder(auto_input)
    reconstructed = decoder(z_out)
    auto = keras.Model(auto_input, reconstructed)
    # KL divergence loss
    kl_loss = -0.5 * tf.reduce_mean(
        1 + z_log_var_out - tf.square(z_mean_out) - tf.exp(z_log_var_out)
    )
    auto.add_loss(kl_loss)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
