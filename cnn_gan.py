# -*- coding: utf-8 -*-
"""CNN_inverter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IkpCQXHhMAp_lnhR8XSjfXmJBS4SDC2x
"""
from __future__ import division, print_function, absolute_import
import time
import numpy as np
import sys
import os
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image
import pdb
import copy
import multiprocessing as mp
import os
import gzip
import struct
import array
import copy
import multiprocessing as mp

from datetime import datetime
# from skimage.measure import compare_ssim
import math
from snwgan import Generator, Discriminator


#GAN Graph Structure
IMG_ROWS = 28
IMG_COLS = 28
NUM_LABEL = 10 
INV_HIDDEN = 5000
NUM_CHANNEL = 1 # Black white image
FILTER_SIZE = 5 # Use 5x5 filter for all conv layer
DEPTH_1 = 32 # num output feature maps for first layer
DEPTH_2 = 64
HIDDEN_UNIT = 1024
CONV_OUT = 7 # convolution output image dimension, calculate based on previous parameters
noise_dim = 128 #! 128 Noise data points
gan_batch_size = 64
L2_REGULAR = 0.01
GAN_CLASS_COE = 10 #!
lam_gp = 10 
gan_learning_rate = 0.00002 #! 0.0002, 0.00005

# Initial training coefficient
EPOCHS = 100
learning_rate = 0.1
loss_beta = 0.003
BATCH_SIZE = 250
ALPHA = 20000
BETA = 5
GAMMA = 1
LAMBDA = 0.06

tf.set_random_seed(1)
tf.reset_default_graph()
def mnist(type):
    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    train_images = parse_images('data/emnist-'+type+'-train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/emnist-'+type+'-train-labels-idx1-ubyte.gz')
    test_images  = parse_images('data/emnist-'+type+'-test-images-idx3-ubyte.gz')
    test_labels  = parse_labels('data/emnist-'+type+'-test-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels


def load_mnist(type):
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist(type)
    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels

def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def inverter(y, model_weights):
  # Input layer
  ww = tf.matmul(model_weights, inv_weights['w_model'])
  wy = tf.matmul(y, inv_weights['w_label'])
  wt = tf.add(wy, ww)
  hidden_layer =  tf.add(wt, inv_weights['b_in'])
  rect = lrelu(hidden_layer, 0.3)
  # Output Layer
  out_layer = tf.add(tf.matmul(rect, inv_weights['w_out']), inv_weights['b_out'])
  rect = lrelu(out_layer, 0.3)
  return rect

#! Loading data
digits_size, digits_x_train, digits_y_train, digits_x_test, digits_y_test = load_mnist('digits')
letters_size, letters_x_train, letters_y_train, letters_x_test, letters_y_test = load_mnist('letters')

#build dataset structure
features = tf.placeholder(tf.float32, shape=[None, IMG_ROWS * IMG_COLS])
labels = tf.placeholder(tf.float32, shape=[None, 10])
batch_size = tf.placeholder(tf.int64)
sample_size = tf.placeholder(tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
dataset = dataset.shuffle(sample_size, reshuffle_each_iteration=True)
dataset = dataset.batch(batch_size, drop_remainder=True).repeat()

iter = dataset.make_initializable_iterator()
next_batch = iter.get_next()

x = tf.placeholder(tf.float32, shape=[None, IMG_ROWS * IMG_COLS])
x_in = tf.reshape(x , [-1,IMG_ROWS, IMG_ROWS,1])
y = tf.placeholder(tf.float32, shape=[None, 10])

# Create CNN weights
conv_w1 = tf.get_variable('cw1', shape=[FILTER_SIZE, FILTER_SIZE, NUM_CHANNEL, DEPTH_1])
conv_w2 = tf.get_variable('cw2', shape=[FILTER_SIZE, FILTER_SIZE, DEPTH_1, DEPTH_2])
conv_b1 = tf.Variable(tf.constant(0.1, shape=[DEPTH_1]), name='b1') # Why initialize to 0.1?
conv_b2 = tf.Variable(tf.constant(0.1, shape=[DEPTH_2]), name='b2')

full_w = tf.get_variable("full_w", [CONV_OUT*CONV_OUT*DEPTH_2, HIDDEN_UNIT])
full_b = tf.Variable(tf.constant(0.1, shape=[HIDDEN_UNIT]), name='full_b')

out_w = tf.get_variable('out_w', [HIDDEN_UNIT,NUM_LABEL])
out_b = tf.Variable(tf.constant(0.1, shape=[NUM_LABEL]), name='out_b')

# Build CNN graph
# First Conv Layer with relu activation and max pool
conv_xw1 = tf.nn.conv2d(x_in,conv_w1,strides=[1, 1, 1, 1], padding='SAME')
conv_z1 = tf.nn.relu(conv_xw1 + conv_b1)
conv_out1 = tf.nn.max_pool(conv_z1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

# Second Conv Layer with relu activation and max pool
conv_xw2 = tf.nn.conv2d(conv_out1, conv_w2,strides=[1, 1, 1, 1], padding='SAME')
conv_z2 = tf.nn.relu(conv_xw2 + conv_b2)
conv_out2 = tf.nn.max_pool(conv_z2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
conv_out2_flat = tf.reshape(conv_out2, [-1, CONV_OUT*CONV_OUT*DEPTH_2])
# Fully Connected Layer with Relu Activation
full_out = tf.nn.relu(tf.matmul(conv_out2_flat, full_w) + full_b)

# Output Layer
y_ml = tf.nn.softmax(tf.matmul(full_out, out_w) + out_b)


#Build Inverter Regularizer
model_weights = tf.concat([tf.reshape(out_w,[1, -1]),tf.reshape(out_b,[1, -1])], 1)


# print(model_weights)
inv_weights = {
  'w_model': tf.get_variable("w_model",[tf.reshape(model_weights, [-1]).shape[0], INV_HIDDEN]),
  'w_label': tf.get_variable("w_label",[NUM_LABEL, INV_HIDDEN]),
  'w_out': tf.get_variable("w_out",[INV_HIDDEN, IMG_ROWS * IMG_COLS]),
  'b_in': tf.Variable(tf.zeros([INV_HIDDEN])),
  'b_out': tf.Variable(tf.zeros([IMG_ROWS * IMG_COLS]))
}

inv_x = inverter(y, model_weights)
#Calculate loss
inv_loss = tf.losses.mean_squared_error(labels=x, predictions=inv_x)
class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_ml))
mi_loss = tf.losses.mean_squared_error(labels=x, predictions=tf.tanh(inv_x))
# calculate prediction accuracy
correct = tf.equal(tf.argmax(y_ml, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

################################################
# Build GAN                                    #
################################################
# Network Inputs
gen_input = tf.placeholder(tf.float32, shape=[gan_batch_size, noise_dim], name='input_noise')
real_input = tf.placeholder(tf.float32, shape=[gan_batch_size, 28*28])
real_input_reshape = tf.reshape(real_input , [-1,28, 28,1])
desired_label = tf.placeholder(tf.float32, shape=[gan_batch_size,10], name='desired_label') #generate image for label 2

# Build Generator Network
gen = Generator(noise_dim, NUM_LABEL, gan_batch_size)
gen_sample = gen(gen_input,desired_label)
gen_sample_reshape = tf.reshape(gen_sample, [gan_batch_size, 28*28])
# Build 2 Discriminator Networks (one from noise input, one from generated samples)
discrim = Discriminator()
disc_real = discrim(real_input_reshape)
disc_fake = discrim(gen_sample)

# Build Loss
def wgan_grad_pen(batch_size,x,G_sample):     
    eps = tf.random_uniform([batch_size,1], minval=0.0,maxval=1.0)
    x_h = eps*x+(1-eps)*G_sample
    x_h = tf.reshape(x_h, [batch_size, 28, 28, 1])
    
    grad_d_x_h = tf.gradients(discrim(x_h), x_h)[0]  
    grad_norm = tf.norm(grad_d_x_h, axis=1, ord='euclidean')
    grad_pen = tf.reduce_mean(tf.square(grad_norm - 1.))
  
    return grad_pen

gan_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=desired_label, logits=y_ml))

#gen_loss = -tf.reduce_mean(tf.log(tf.maximum(0.001, disc_fake)))# + GAN_CLASS_COE*gan_class_loss
#disc_loss = -tf.reduce_mean(tf.log(tf.maximum(0.001, disc_real)) + tf.log(tf.maximum(0.001, 1. - disc_fake))) + lam_gp*wgan_grad_pen(gan_batch_size,real_input, gen_sample_reshape)

#Improved WGAN with gradient penalty
gen_loss = -tf.reduce_mean(disc_fake) + GAN_CLASS_COE*gan_class_loss
disc_loss = -tf.reduce_mean(disc_real) + tf.reduce_mean(disc_fake) + lam_gp*wgan_grad_pen(gan_batch_size,real_input, gen_sample_reshape)

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=gan_learning_rate, beta1=0.5, beta2=0.999) #! 0.999
optimizer_disc = tf.train.AdamOptimizer(learning_rate=gan_learning_rate, beta1=0.5, beta2=0.999)

# Training Variables for each optimizer
# gen Network Variables
gen_vars = [gen.linear_w, gen.linear_b, gen.deconv_w1, gen.deconv_w2, gen.deconv_w3] 
            # gen.deconv_b1, gen.deconv_b2, gen.deconv_b3]

# Discriminator Network Variables
disc_vars = [discrim.conv_w1, discrim.conv_w2, discrim.conv_w3, discrim.conv_w4, discrim.conv_w5,
            # discrim.conv_b1, discrim.conv_b2, discrim.conv_b3, discrim.conv_b4, discrim.conv_b5,
            discrim.linear_w1, discrim.linear_b1, discrim.linear_w2, discrim.linear_b2]

# Create training operations
train_GAN = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

def train(loss_beta, learning_rate, Epoch, Batch):
  #!
  # total_loss = class_loss - loss_beta * mi_loss
  total_loss = class_loss
  steps_per_epoch = int(digits_size/ BATCH_SIZE)
  global_step = tf.train.get_or_create_global_step()
  learn_rate = tf.train.exponential_decay(1e-3, global_step, decay_steps=2*steps_per_epoch, decay_rate=0.97, staircase=True)
  model_optimizer = tf.train.AdamOptimizer(learn_rate).minimize(total_loss, var_list=[conv_w1, conv_w2, conv_b1, conv_b2, full_w, full_b, out_w, out_b])
  #!
  # inverter_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(inv_loss)
  init_vars = tf.global_variables_initializer()
  
  with tf.Session() as sess:
    sess.run(init_vars)
   
    # initialise iterator with train data
    sess.run(iter.initializer, feed_dict = {features: digits_x_train, labels: digits_y_train, batch_size: Batch, sample_size: 70000})
    
    print('Beta %g Training...'%(loss_beta))
    for i in range(Epoch):
      for step in range(steps_per_epoch):
        batch = sess.run(next_batch)
        model_optimizer.run(feed_dict={ x: batch[0], y: batch[1]})
        #inverter_optimizer.run(feed_dict={ x: batch[0], y: batch[1]})
      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1] })
      print('Epoch %d, training accuracy %g' % (i, train_accuracy))    
    
    # initialise iterator with test data
    sess.run(iter.initializer, feed_dict = {features: digits_x_test, labels: digits_y_test, batch_size: digits_y_test.shape[0], sample_size: 1})
    batch = sess.run(next_batch)
    test_acc = accuracy.eval(feed_dict={x: batch[0], y: batch[1] })
    print("beta is %g, test accuracy is %g"%(loss_beta, test_acc))
    
    train_GAN_MI(sess, 150) 
    return test_acc

def plot_gan_image(name, epoch, sess):
  #Finish Training the GAN
  gen.training = False
  discrim.training = False

  # Generate images from noise, using the generator network.
  f, a = plt.subplots(5, 10, figsize=(10, 4))
  for i in range(10):
      # Desired label
      d_label = np.zeros([gan_batch_size, 10])
      d_label[:,i] = 1
      # Noise input.
      z = np.random.uniform(-1., 1., size=[gan_batch_size, noise_dim])
      g = sess.run([gen_sample], feed_dict={gen_input: z, desired_label: d_label})
      g = np.array(g)
      g = np.reshape(g, newshape=(gan_batch_size, 28, 28, 1))
      #print the top 5 images
      g = g[:5,:]

      # Reverse colours for better display
      #g = -1 * (g - 1)
      for j in range(5):
          # Generate image from noise. Extend to 3 channels for matplot figure.
          img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                           newshape=(28, 28, 3))
          a[j][i].imshow(img, origin='lower')

  # f.show()
  plt.draw()
  plt.savefig(name+epoch)
  #Continue Training
  gen.training = True
  discrim.training = True 
  
def train_GAN_MI(sess, Epoch):  
  steps_per_epoch = int(letters_size/ gan_batch_size)

  #Update Mean and Variance of batch normalization during traininEpoch %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))g
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  SPECTRAL_NORM_UPDATE_OPS = "spectral_norm_update_ops"
  spectral_norm_update_ops = tf.get_collection(SPECTRAL_NORM_UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    # initialise iterator with letters train data
    sess.run(iter.initializer, feed_dict = {features: letters_x_train, labels: letters_y_train, batch_size: gan_batch_size, sample_size: 10000})

    for i in range(0, Epoch):
      for step in range(0, steps_per_epoch):
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, batch_y = sess.run(next_batch)
        # Use one hot aux_label instead of distribution
        aux_label = sess.run(y_ml, feed_dict={x: batch_x})
        # aux_label = tf.one_hot(tf.argmax(aux_label, axis=1), 10)
        # aux_label = aux_label.eval()
        #Just train for image 3
        # aux_label = np.zeros([gan_batch_size, 10])
        # aux_label[:,3] = 1

        # Sample random noise 
        z = np.random.uniform(-1., 1., size=[gan_batch_size, noise_dim])
        # Pass random noise into the generator and get generated image
        gen_mi = sess.run(gen_sample, feed_dict={gen_input:z, desired_label: aux_label}) 
        gen_mi = np.reshape(gen_mi, [gan_batch_size, 28*28])

        #! Train Discriminator
        # if step % 5 == 0:
        train_disc.run(feed_dict={real_input: batch_x,  gen_input: z, x: gen_mi, desired_label: aux_label})

        #Train Generator
        if step % 5 == 0:
          train_GAN.run(feed_dict={real_input: batch_x,  gen_input: z, x: gen_mi, desired_label: aux_label})

        for update_op in spectral_norm_update_ops:
          sess.run(update_op)

      gl,dl = sess.run([gen_loss, disc_loss], feed_dict={real_input: batch_x,  gen_input: z, x: gen_mi, desired_label: aux_label})
      print('Epoch %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
      #plot the gan image for every 2 epoch
      # if i % 5 == 0:
      plot_gan_image('cnn_diffdt_noinv_wasser_5dis_epoch',str(i), sess)       

#Will not run when file is imported by other files
if __name__ == '__main__':
  # !acc = train(0.001, 0.1, 3, 200)
  acc = train(0.001, 0.1, 30, 200)
