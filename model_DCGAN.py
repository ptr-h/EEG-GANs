# Acknowledgments: 
# https://www.tensorflow.org/tutorials/generative/dcgan
# https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy 
from tensorflow.keras import backend as K
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM

class DCGAN(): 
  def __init__(self, gen_optim, disc_optim,input_dim, noise_dim=100,dropout=0.2):
    
    # setup config variables eg. noise_dim, hyperparams, verbose, plotting etc. 
    self.noise_dim = noise_dim
    self.dropout = dropout
    self.input_dim = input_dim

    # Build and compile the discriminator
    self.discriminator = self.build_discriminator()
    # Ensure discriminator is trainable 
    self.discriminator.compile(loss='categorical_crossentropy',
            optimizer= disc_optim,
            metrics=['accuracy'])

    # Build the generator
    self.generator = self.build_generator()

    # The generator takes noise as input and generates eeg data
    self.combined = self.build_GAN()
    self.combined.compile(loss='binary_crossentropy',
                          optimizer=gen_optim)

    # history variables
    self.loss_history, self.acc_history, self.grads_history = {}, {}, {}

  def build_generator(self):
    model = Sequential()
    model.add(layers.Dense(128, use_bias=False, input_shape=(self.noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Dense(256, use_bias=False, input_shape=(128, )))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Dense(512, use_bias=False, input_shape=(256,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Dense(256, use_bias=False, input_shape=(512,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Dense(self.input_dim, use_bias=False,activation="tanh", input_shape=(256,)))
    model.add(layers.Reshape((1, self.input_dim)))
    



    noise = layers.Input(shape=(self.noise_dim,))
    signal = model(noise)
    
    return Model(noise, signal)

  def build_discriminator(self):
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, self.input_dim), activation=tf.keras.layers.LeakyReLU(alpha=0.3), return_sequences=True))
    model.add(layers.Dropout(self.dropout))
    model.add(LSTM(32, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
    model.add(layers.Dropout(self.dropout))
    model.add(layers.Dense(1, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))

    signal = layers.Input(shape=(1,self.input_dim))
    validity = model(signal)
    
    return Model(signal, validity)
  
  def build_GAN(self): 
    # Generator takes noise and outputs generated eeg data
    z = layers.Input(shape=(self.noise_dim,))
    generated_eeg = self.generator(z)
    
    # For the combined model we will only train the generator
    discriminator = self.discriminator
    discriminator.trainable = False

    # The discriminator takes generated eeg data as input and determines validity
    validity = discriminator(generated_eeg)

    return Model(z,validity)

  # generate fake data! 
  def generate_fake_data(self,N=100): 
    noise = np.random.normal(0, 1, (N, self.noise_dim))
    gen_signal = self.generator.predict(noise)
    return gen_signal, noise
  
  # training loop
  def train(self, train_dataset, epochs=500, batch_size=32,discriminator_iters=1,label_smoothing=0,plot=False):
    '''
    Training loop
    INPUTS: 
    train_dataset - EEG training dataset as numpy array with shape=(trials,eeg,freq_bins,time_bins)
              Assumed dataset has already been normalized! 
    epochs - 
    batch_size - 
    plot - 
    '''
    # init loss history params 
    loss_history, acc_history, grads_history = self.loss_history, self.acc_history, self.grads_history
    gen_grads_history, disc_grads_history, real_grads_history, fake_grads_history = [], [], [], []
    gen_loss_history, disc_loss_history, real_loss_history, fake_loss_history = [], [], [], []
    gen_acc_history, disc_acc_history, real_acc_history, fake_acc_history = [], [], [], []

    # init training dataset that can be shuffled
    X_train = train_dataset.astype('float32') 
  
    for epoch in range(epochs):
      start = time.time()
      
      # shuffle training dataset 
      np.random.shuffle(X_train)

      # batch useful variables 
      num_batches = int(np.ceil(X_train.shape[0] / float(batch_size)))
      
      # grad, loss and acc parameters
      grads_real_l2_norm, grads_fake_l2_norm, grads_disc_l2_norm, grads_gen_l2_norm = 0,0,0,0
      d_loss, d_loss_real, d_loss_fake, g_loss = 0,0,0,0
      d_acc, d_acc_real, d_acc_fake, g_acc = 0,0,0,0

      with tqdm(range(num_batches),unit="batch") as tepoch:
        for batch in tepoch:
          tepoch.set_description(f"Epoch {epoch+1}")
        
          # final batch
          if batch==num_batches-1:
            eeg_data = X_train[batch*batch_size:]
          else:
            eeg_data = X_train[batch*batch_size:(batch+1)*batch_size]

          # ---------------------
          #  Train Discriminator
          # ---------------------

          assert discriminator_iters > 0, 'Number of discriminator must be positive integer'
          for _ in range(discriminator_iters):
            # Generate batch of fake eeg data for discriminator to train on
            gen_signal, noise = self.generate_fake_data(N=eeg_data.shape[0])

            # label smoothing
            fake = np.zeros((eeg_data.shape[0],1)) + 0.5 * label_smoothing
            valid = np.ones((eeg_data.shape[0],1)) * (1.0 - label_smoothing) + 0.5 * label_smoothing

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real_batch, d_acc_real_batch = self.discriminator.train_on_batch(eeg_data, valid)
            d_loss_fake_batch, d_acc_fake_batch = self.discriminator.train_on_batch(gen_signal, fake)

            # get discriminator gradients at input w/ real and fake data
            inp_real = tf.Variable(eeg_data,dtype='float32')
            with tf.GradientTape() as tape:
              pred_real = self.discriminator(inp_real)
            grads_real = tape.gradient(pred_real, inp_real).numpy()

            inp_fake = tf.Variable(gen_signal,dtype='float32')
            with tf.GradientTape() as tape:
              pred_fake = self.discriminator(inp_fake)

            grads_fake = tape.gradient(pred_fake, inp_fake).numpy()

            # update grad, loss and acc tracking
            grads_real_l2_norm += np.sqrt(np.sum(np.square(grads_real)))/(float(num_batches)*discriminator_iters)
            grads_fake_l2_norm += np.sqrt(np.sum(np.square(grads_fake)))/(float(num_batches)*discriminator_iters)
            grads_disc_l2_norm += 0.5 * (grads_fake_l2_norm + grads_real_l2_norm)/(float(num_batches)*discriminator_iters)
            d_loss_real += d_loss_real_batch/(float(num_batches)*discriminator_iters)
            d_acc_real += d_acc_real_batch/(float(num_batches)*discriminator_iters)
            d_loss_fake += d_loss_fake_batch/(float(num_batches)*discriminator_iters)
            d_acc_fake += d_acc_fake_batch/(float(num_batches)*discriminator_iters)
            d_loss_batch = 0.5 * (d_loss_real_batch + d_loss_fake_batch)
            d_acc_batch = 0.5 * (d_acc_real_batch + d_acc_fake_batch)
            d_loss += d_loss_batch/(float(num_batches)*discriminator_iters)
            d_acc += d_acc_batch/(float(num_batches)*discriminator_iters)

                
          # ---------------------
          #  Train Generator
          # ---------------------
          # Generate 2*batch of fake eeg data for generator to train on
          gen_signal, noise = self.generate_fake_data(N=2*eeg_data.shape[0])
          valid = np.ones((2*eeg_data.shape[0],1))

          # Train the generator (wants discriminator to mistake images as real)
          g_loss_batch = self.combined.train_on_batch(noise, valid)
          # Manually calculate accuracy to avoid dropout layer
          g_acc_batch = np.average(np.round(self.combined.predict(noise)))

          # get generator gradients at input
          inp_noise = tf.Variable(np.random.normal(0, 1, (eeg_data.shape[0], self.noise_dim)),dtype='float32')
          with tf.GradientTape() as tape:
            pred = self.combined(inp_noise)

          grads = tape.gradient(pred, inp_noise).numpy()

          # update grad, loss and acc tracking
          grads_gen_l2_norm += np.sqrt(np.sum(np.square(grads)))/float(num_batches)
          g_loss += g_loss_batch/float(num_batches)
          g_acc += g_acc_batch/float(num_batches)


          # ---------------------
          # Debugging
          # ---------------------

          # print('Combined GAN batch acc: {}%'.format(100*np.average(np.round(self.combined.predict(noise)))))

          # print('Disc grads: real= {}, fake={}, avg= {}'.format(grads_real_l2_norm,grads_fake_l2_norm,grads_disc_l2_norm))

          # print('Gen grads: {}'.format(grads_gen_l2_norm))

      # Save the grad, loss and accuracy histories
      gen_grads_history.append(grads_gen_l2_norm) 
      disc_grads_history.append(grads_disc_l2_norm) 
      real_grads_history.append(grads_real_l2_norm) 
      fake_grads_history.append(grads_fake_l2_norm) 
      gen_loss_history.append(g_loss) 
      disc_loss_history.append(d_loss) 
      real_loss_history.append(d_loss_real) 
      fake_loss_history.append(d_loss_fake) 
      gen_acc_history.append(g_acc)
      disc_acc_history.append(d_acc) 
      real_acc_history.append(d_acc_real) 
      fake_acc_history.append(d_acc_fake) 

      # Plot the progress
      print ('Epoch #: {}/{}, time taken: {} secs \n'.format(epoch+1,epochs,time.time()-start))
      print('Disc:  loss= {}, acc w/ dropout= {}%, grads= {} \n'.format(d_loss,100*d_acc,grads_disc_l2_norm))
      print('Disc Fake:  loss= {}, acc w/ dropout= {}%, grads= {} \n'.format(d_loss_fake,100*d_acc_fake,grads_fake_l2_norm))
      print('Disc Real:  loss= {}, acc w/ dropout= {}%, grads= {} \n'.format(d_loss_real,100*d_acc_real,grads_real_l2_norm))
      print('Gen:  loss= {}, acc w/o dropout= {}%, grads= {} \n'.format(g_loss,100*g_acc,grads_gen_l2_norm))
      
      if plot: 
        # fake image example
        generated_image,_ = self.generate_fake_data(N=1)       
        # real image example  
        trial_ind, eeg = 0, 0
        real_image = np.expand_dims(train_dataset[trial_ind], axis=0)

        # plot discriminator classification
        fake_predictions = self.discriminator.predict(self.generate_fake_data(N=train_dataset.shape[0]))
        real_predictions = self.discriminator.predict(train_dataset)
        plt.figure()
        plt.plot(real_predictions,'bo')
        plt.plot(fake_predictions,'ro')
        plt.legend(['Real', 'Fake'])
        plt.show()

    
    # plot loss history
    plt.figure()
    plt.plot(gen_loss_history, 'r')
    plt.plot(disc_loss_history, 'b')
    plt.plot(real_loss_history, 'g')
    plt.plot(fake_loss_history, 'k')
    plt.title('Loss history')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Generator', 'Discriminator', 'Real', 'Fake'])

    # plot accuracy history
    plt.figure()
    plt.plot(100*gen_acc_history, 'r')
    plt.plot(100*disc_acc_history, 'b')
    plt.plot(100*real_acc_history, 'g')
    plt.plot(100*fake_acc_history, 'k')
    plt.title('Accuracy history')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend(['Generator', 'Discriminator', 'Real', 'Fake'])

    # plot grads history
    plt.figure()
    plt.plot(gen_grads_history, 'r')
    plt.plot(disc_grads_history, 'b')
    plt.plot(real_grads_history, 'g')
    plt.plot(fake_grads_history, 'k')
    plt.title('L2-norm of Gradients at input history')
    plt.xlabel('Epochs')
    plt.ylabel('L2-norm of Gradients')
    plt.legend(['Generator', 'Discriminator', 'Real', 'Fake'])
    
    grads_history['Gen'], grads_history['Disc'] = gen_grads_history, disc_grads_history
    grads_history['Real'], grads_history['Fake'] = real_grads_history, fake_grads_history

    loss_history['Gen'], loss_history['Disc'] = gen_loss_history, disc_loss_history
    loss_history['Real'], loss_history['Fake'] = real_loss_history, fake_loss_history
    
    acc_history['Gen'], acc_history['Disc'] = gen_acc_history, disc_acc_history
    acc_history['Real'], acc_history['Fake'] = real_acc_history, fake_acc_history
    

    self.loss_history, self.acc_history = loss_history, acc_history
    
    return loss_history, acc_history, grads_history
