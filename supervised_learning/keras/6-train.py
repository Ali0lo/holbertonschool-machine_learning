#!/usr/bin/env python3
"""Train a keras model with optional early stopping"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains the model using mini-batch gradient descent,
    optionally using early stopping based on validation loss
    """

    callbacks = []

    # Only add early stopping if requested AND validation data is provided
    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',   # monitor validation loss
            patience=patience,
            restore_best_weights=True  # restore model weights from best epoch
        )
        callbacks.append(early_stop)

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )

    return history
