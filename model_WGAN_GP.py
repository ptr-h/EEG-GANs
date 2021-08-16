# Acknowledgements:
# https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras import backend as K
from tqdm import tqdm_notebook
from IPython.display import clear_output
from tensorflow.keras.layers import LSTM
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm


class WGAN_GP():
    def __init__(self, gen_optimizer, disc_optimizer, input_dim, noise_dim=100, dropout=0.2):

        # setup config variables eg. noise_dim, hyperparams, verbose, plotting etc.
        self.noise_dim = noise_dim
        self.dropout = dropout
        self.input_dim = input_dim

        self.generator_optimizer = gen_optimizer
        self.discriminator_optimizer = disc_optimizer

        # setup history dictionary
        self.history = {}

        # build discriminator and generator models
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        model = Sequential()
        model.add(layers.Dense(128, use_bias=False, input_shape=(self.noise_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Dense(256, use_bias=False, input_shape=(128,)))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Dense(512, use_bias=False, input_shape=(256,)))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Dense(256, use_bias=False, input_shape=(512,)))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Dense(self.input_dim, use_bias=False, input_shape=(256,)))
        model.add(layers.Reshape((1, self.input_dim)))

        noise = layers.Input(shape=(self.noise_dim,))
        signal = model(noise)

        return Model(noise, signal)

    def build_discriminator(self):
        model = Sequential()
        model.add(LSTM(64, input_shape=(1, self.input_dim), activation="relu", return_sequences=True))
        model.add(layers.Dropout(self.dropout))
        model.add(LSTM(32, activation="relu"))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(1, activation="linear"))

        signal = layers.Input(shape=(1, self.input_dim))
        validity = model(signal)

        return Model(signal, validity)

    def build_GAN(self):
        # Generator takes noise and outputs generated eeg data
        z = layers.Input(shape=(self.noise_dim,))
        generated_eeg = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated eeg data as input and determines validity
        validity = self.discriminator(generated_eeg)

        return Model(z, validity)

    # generate fake data after training!
    def generate_fake_data(self, N=100):
        noise = tf.random.normal([N, self.noise_dim]).numpy()
        return self.generator(noise, training=False).numpy(), noise

        # loss functions

    def disc_loss(self, fake_logits, real_logits):
        return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

    def gen_loss(self, fake_logits):
        return - tf.reduce_mean(fake_logits)

    # gradient penalty term for discriminator
    def gradient_penalty(self, discriminator, real_data, gen_signal):
        eps = tf.random.uniform([real_data.shape[0], 1, 1], 0., 1.)
        inter = real_data + (eps * (real_data - gen_signal))
        with tf.GradientTape() as tape:
            tape.watch(inter)
            pred = discriminator(inter)

        grad = tape.gradient(pred, inter)[0]
        grad_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(grad)))

        return tf.reduce_mean((grad_l2_norm-1.0)**2)

        # training functions

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images):

        # loss variables to return
        disc_loss, disc_grads = 0, 0

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # train discriminator over several iterations
        for _ in range(self.discriminator_iters):
            # setup gradient tools -- GradientTape automatically watches all trainable variables
            with tf.GradientTape() as disc_tape:
                # forward prop
                noise = tf.random.normal([images.shape[0], self.noise_dim])
                gen_signal = self.generator(noise, training=True)
                fake_logits = self.discriminator(gen_signal, training=True)
                real_logits = self.discriminator(images, training=True)

                # calculate loss
                loss = self.disc_loss(fake_logits, real_logits)
                gp = self.gradient_penalty(partial(self.discriminator, training=True), images, gen_signal)
                loss += self.gp_weight * gp

                # back prop
            disc_grads = disc_tape.gradient(loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

            # save some variables for history
            disc_loss += loss
            disc_grads += disc_grads

        # ---------------------
        #  Train Generator
        # ---------------------
        noise = tf.random.normal([images.shape[0], self.noise_dim])
        with tf.GradientTape() as gen_tape:
            gen_signal = self.generator(noise, training=True)
            fake_logits = self.discriminator(gen_signal, training=True)
            gen_loss = self.gen_loss(fake_logits)

        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        return disc_loss, disc_grads[0], gen_loss, gen_grads[0]

    # training loop
    def train(self, train_dataset,
              epochs=25, batch_size=32, discriminator_iters=5,
              gp_weight=0, plot=False, save_plots=False):

        # set up data for training
        dataset = tf.data.Dataset.from_tensor_slices(train_dataset.astype('float32')).shuffle(
            train_dataset.shape[0]).batch(
            batch_size)
        N_batch = np.ceil(train_dataset.shape[0] / float(batch_size))

        # save training variables
        self.discriminator_iters = discriminator_iters
        self.gp_weight = gp_weight

        # setup history variables
        history = self.history
        history['grads'], history['loss'], history['acc'] = {}, {}, {}
        gen_loss_history, disc_loss_history = [], []
        gen_grads_history, disc_grads_history = [], []
        gen_acc_history, disc_acc_history = [], []

        # start training loop
        for epoch in range(epochs):
            start = time.time()

            # refresh loss for every epoch
            gen_loss, disc_loss, disc_grads, gen_grads = 0, 0, 0, 0

            with tqdm(total=N_batch, position=0, leave=True) as pbar:
                for image_batch in dataset:
                    # train step
                    disc_loss_batch, disc_grads_batch, gen_loss_batch, gen_grads_batch = self.train_step(image_batch)

                    # convert variables to usable format
                    disc_loss_batch = tf.reduce_mean(disc_loss_batch).numpy() / float(self.discriminator_iters)
                    disc_grads_batch = tf.reduce_mean(
                        tf.sqrt(tf.reduce_sum(tf.square(disc_grads_batch)))).numpy() / float(
                        self.discriminator_iters)
                    gen_loss_batch = tf.reduce_mean(gen_loss_batch).numpy()
                    gen_grads_batch = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(gen_grads_batch)))).numpy()

                    # store history
                    gen_loss += gen_loss_batch / float(N_batch)
                    disc_loss += disc_loss_batch / float(N_batch)
                    gen_grads += gen_grads_batch / float(N_batch)
                    disc_grads += disc_grads_batch / float(N_batch)

                    pbar.update()
            pbar.close()

            # store history
            gen_loss_history.append(gen_loss)
            disc_loss_history.append(disc_loss)
            gen_grads_history.append(gen_grads)
            disc_grads_history.append(disc_grads)

            print('Epoch #: {}/{}, Time taken: {} secs,\n Grads: disc= {}, gen= {},\n Losses: disc= {}, gen= {}' \
                  .format(epoch + 1, epochs, time.time() - start, disc_grads, gen_grads, disc_loss, gen_loss))

        # Generate after the final epoch
        clear_output(wait=True)

        plt.figure()
        plt.plot(gen_loss_history, 'r')
        plt.plot(disc_loss_history, 'b')
        plt.title('Loss history')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Generator', 'Discriminator'])
        plt.show()

        plt.figure()
        plt.plot(gen_grads_history, 'r')
        plt.plot(disc_grads_history, 'b')
        plt.title('Gradient history')
        plt.xlabel('Epochs')
        plt.ylabel('Gradients (L2 norm)')
        plt.legend(['Generator', 'Discriminator'])
        plt.show()

        history['grads']['gen'], history['grads']['disc'] = gen_grads_history, disc_grads_history
        history['loss']['gen'], history['loss']['disc'] = gen_loss_history, disc_loss_history

        self.history = history

        return history
