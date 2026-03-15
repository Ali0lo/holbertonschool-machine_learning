#!/usr/bin/env python3
"""Trains a Keras model with validation, early stopping, and LR decay."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent."""
    callbacks = []

    if early_stopping is True and validation_data is not None:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        ))

    if learning_rate_decay is True and validation_data is not None:
        def schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearningRateScheduler(
            schedule=schedule,
            verbose=1
        ))

    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )
