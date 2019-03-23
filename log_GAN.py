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

from datetime import datetime
from skimage.measure import compare_ssim
from tensorflow.examples.tutorials.mnist import input_data
import math
from gan_mi import Generator, Discriminator



#Graph Structure
IMG_ROWS = 28
IMG_COLS = 28
NUM_LABEL = 10 # using half as train, half as auxiliary
INV_HIDDEN = 5000
NUM_CHANNEL = 1 # Black white image
FILTER_SIZE = 5 # Use 5x5 filter for all conv layer
DEPTH_1 = 32 # num output feature maps for first layer
DEPTH_2 = 64
HIDDEN_UNIT = 1024
CONV_OUT = 7 # convolution output image dimension, calculate based on previous parameters
noise_dim = 10 # Noise data points
gan_batch_size = 200

# Initial training coefficient
EPOCHS = 100
learning_rate = 0.1
gan_learning_rate = 0.0002
loss_beta = 0.003
BATCH_SIZE = 250
ALPHA = 20000
BETA = 5
GAMMA = 1
LAMBDA = 0.06

tf.reset_default_graph()
tf.set_random_seed(1)


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, IMG_ROWS * IMG_COLS])
y = tf.placeholder(tf.float32, shape=[None, 10])


def layer(input, num_units):
  W = tf.Variable(tf.zeros([input.shape[1], num_units], tf.float32), name="w")
  B = tf.Variable(tf.zeros([num_units], tf.float32), name="b")
  output = tf.nn.bias_add(tf.matmul(input,W), B)
  return W, B, output

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

#Build Logistic Layer
with tf.name_scope("logistic_layer"):
  w,b,z = layer(x,NUM_LABEL)
  y_ml = tf.nn.softmax(z)

#Build Inverter Regularizer
model_weights = tf.concat([tf.reshape(w,[1, -1]),tf.reshape(b,[1, -1])], 1)
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
class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_ml))
mi_loss = tf.losses.mean_squared_error(labels=x, predictions=tf.tanh(inv_x))
# calculate prediction accuracy
correct = tf.equal(tf.argmax(y_ml, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

################################################
# Build GAN                                    #
################################################
# Network Inputs
gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
disc_input = tf.placeholder(tf.float32, shape=[None, 28*28])
disc_input_reshape = tf.reshape(disc_input , [-1,28, 28,1])
desired_label = tf.placeholder(tf.float32, shape=[None,10], name='desired_label') #generate image for label 2
# Build Generator Network
gen = Generator(noise_dim, NUM_LABEL, gan_batch_size)
gen_sample = gen.build(gen_input,desired_label)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
discrim = Discriminator()
disc_real = discrim.build(disc_input_reshape)
disc_fake = discrim.build(gen_sample)

# Build Loss
gan_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=desired_label, logits=y_ml))
# real_loss = -tf.reduce_mean(tf.log(disc_real))
# fake_loss = -tf.reduce_mean(tf.log(disc_fake))
gen_loss = -tf.reduce_mean(tf.log(tf.maximum(0.001, disc_fake))) + gan_class_loss
disc_loss = -tf.reduce_mean(tf.log(tf.maximum(0.001, disc_real)) + tf.log(tf.maximum(0.001, 1. - disc_fake)))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=gan_learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=gan_learning_rate)

# Training Variables for each optimizer
# gen Network Variables
gen_vars = [gen.linear_w, gen.linear_b, gen.deconv_w1, gen.deconv_w2, gen.deconv_w3, 
            gen.deconv_b1, gen.deconv_b2, gen.deconv_b3]
# Discriminator Network Variables
disc_vars = [discrim.conv_w1, discrim.conv_w2, discrim.conv_w3, discrim.conv_w4, discrim.conv_w5,
            discrim.conv_b1, discrim.conv_b2, discrim.conv_b3, discrim.conv_b4, discrim.conv_b5,
            discrim.linear_w, discrim.linear_b]

# Create training operations
train_GAN = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

def train(loss_beta, learning_rate, Epoch, Batch):
#   total_loss = class_loss - loss_beta * mi_loss
  total_loss = class_loss
  model_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, var_list=[w,b])
#   inverter_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(inv_loss)
  init_vars = tf.global_variables_initializer()
  
  with tf.Session() as sess:
    sess.run(init_vars)
   
    print('Beta %g rate %g batch %g Training...'%(loss_beta, learning_rate, Batch))
    for i in range(Epoch):
      batch = mnist.train.next_batch(Batch)
      model_optimizer.run(feed_dict={ x: batch[0], y: batch[1]})
#       inverter_optimizer.run(feed_dict={ x: batch[0], y: batch[1]})
      if i % 1000 == 0:  
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1] })
        valid_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, y: mnist.validation.labels })
        print('step %d, training accuracy %g, validation accuracy %g' % (i, train_accuracy,valid_accuracy))    
    
    test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels })
    print("beta is %g, test accuracy is %g"%(loss_beta, test_acc))
    
    train_GAN_MI(sess, 20000) 

    return test_acc

def show_image(array):
  adv_img = plt.imshow(np.reshape(array, (28, 28)), cmap="gray", vmin=array.min(), vmax=array.max())
  plt.show(adv_img)

def train_GAN_MI(sess, num_steps):
  d_label = np.zeros([gan_batch_size, 10])
  d_label[:,2] = 1

  for i in range(1, num_steps+1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    batch_x, batch_y = mnist.train.next_batch(gan_batch_size)

    z = np.random.uniform(-1., 1., size=[gan_batch_size, noise_dim])
    # Train
    gen_mi = sess.run(gen_sample, feed_dict={gen_input:z, desired_label: d_label}) 
    gen_mi = np.reshape(gen_mi, [gan_batch_size, 28*28])

    # print('disc_input',disc_input.shape, 'batch_x',batch_x.shape, 'x',x.shape,  'gen_mi', gen_mi.shape)
    _, _, gl, dl = sess.run([train_disc, train_GAN, gen_loss, disc_loss],
                            feed_dict={disc_input: batch_x,  gen_input: z, x: gen_mi, desired_label: d_label})

    #train one generator for every 5 discriminator
    # if i % 2 == 0:
    #   _, gl, dl= sess.run([train_GAN, gen_loss, disc_loss],
    #                         feed_dict={disc_input: batch_x,  gen_input: z, x: gen_mi, desired_label: d_label})

    if i % 100 == 0 or i == 1:
      print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
            
  # Generate images from noise, using the generator network.
  f, a = plt.subplots(4, 10, figsize=(10, 4))
  for i in range(10):
      # Noise input.
      z = np.random.uniform(-1., 1., size=[gan_batch_size, noise_dim])
      d_label = np.zeros([gan_batch_size, 10])
      d_label[:,2] = 1
      g = sess.run([gen_sample], feed_dict={gen_input: z, desired_label: d_label})
      g = np.array(g)
      g = np.reshape(g, newshape=(gan_batch_size, 28, 28, 1))
      #print the top 4 images
      g = g[:4,:]

      # Reverse colours for better display
      g = -1 * (g - 1)
      for j in range(4):
          # Generate image from noise. Extend to 3 channels for matplot figure.
          img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                           newshape=(28, 28, 3))
          a[j][i].imshow(img)

  # f.show()
  plt.draw()
  plt.savefig('GAN_MI')

#Will not run when file is imported by other files
if __name__ == '__main__':
  acc = train(0.001, 0.1, 8000, 250)
