# Simple GAN

Implementation of a simple Generative Adversarial Network (GAN) using TensorFlow/Keras.

## Files

- 0-simple_gan.py

## Class

### Simple_GAN

Methods:

- __init__
- get_real_sample
- get_fake_sample
- train_step

## Training Strategy

1. Train discriminator `disc_iter` times.
2. Compute discriminator loss on:
   - real samples -> target = 1
   - fake samples -> target = -1
3. Update discriminator weights.
4. Train generator once.
5. Generator objective:
   - discriminator(generator(z)) -> target = 1

## Dependencies

- tensorflow
- numpy
- matplotlib
