
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Training Params
num_steps = 100000
learning_rate = 0.0002
x_dim = 2
noise_dim = 2
NUM_LABEL = 2
GAN_CLASS_COE = 1
# Network Params
# noise_dim = 100 # Noise data points
# NUM_LABEL = 10 #10 labels

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))
 
# Linear Regression 
class Generator(object):
  # Generator Parameters
  def __init__(self, noise_dim, NUM_LABEL, batch_size):
    self.batch_size = batch_size
    self.linear_w1 = tf.Variable(glorot_init([noise_dim+NUM_LABEL, 20]),name='glw')
    self.linear_b1 = tf.Variable(glorot_init([20]),name='glb')
    self.linear_w2 = tf.Variable(glorot_init([20, x_dim]),name='glw')
    self.linear_b2 = tf.Variable(glorot_init([x_dim]),name='glb')
    
    self.training = True

 # Build Generator Graph
  def __call__(self, z,y):
    z_y = tf.concat([z,y],1)
    linear_z1 = tf.nn.leaky_relu(tf.matmul(z_y,self.linear_w1)+self.linear_b1)
    out_layer = tf.matmul(z_y,self.linear_w2)+self.linear_b2
    return out_layer

# Logistic Regression
class Discriminator(object):
  # Discriminator Parameters
  def __init__(self):
    self.linear_w1 = tf.Variable(glorot_init([x_dim, 20]))
    self.linear_b1 = tf.Variable(glorot_init([20]))
    self.linear_w2 = tf.Variable(glorot_init([20, 1]))
    self.linear_b2 = tf.Variable(glorot_init([1]))

    self.training = True

  # Build Discriminator Graph
  def __call__(self, x):
    linear1 = tf.nn.leaky_relu(tf.matmul(x, self.linear_w1) + self.linear_b1)
    out = tf.matmul(linear1, self.linear_w2) + self.linear_b2
    out = tf.sigmoid(out)
    return out

class Classifier(object):
  def __init__(self):
    self.linear_w1 = tf.Variable(glorot_init([x_dim, 20]))
    self.linear_b1 = tf.Variable(glorot_init([20]))
    self.linear_w2 = tf.Variable(glorot_init([20, 1]))
    self.linear_b2 = tf.Variable(glorot_init([1]))

    self.training = True

  # Build Discriminator Graph
  def __call__(self, x):
    linear1 = tf.nn.leaky_relu(tf.matmul(x, self.linear_w1) + self.linear_b1)
    out = tf.matmul(linear1, self.linear_w2) + self.linear_b2
    out = tf.sigmoid(out)
    return out

def train_gan(sess):
  # mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
  tf.reset_default_graph()
  tf.set_random_seed(1)

  # Train Classifier
  x_1 = np.random.normal(loc=mu1, scale=sigma1, size=10000)
  x_2 = np.random.normal(loc=mu2, scale=sigma2, size=10000)
  y_1 = np.zeros(10000)
  y_2 = np.ones(10000)
  train_x = np.concatenate((x_1, x_2), axis=0)
  train_y = np.concatenate((y_1, y_2), axis=0)   

  # Build Networks
  # Network Inputs
  gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
  x = tf.placeholder(tf.float32, shape=[None, x_dim])
  label_input = tf.placeholder(tf.float32, shape=[None, NUM_LABEL], name='label_input')
  # Build Generator Network
  generator = Generator()
  gen_sample = generator(gen_input,label_input)

  # Build 2 Discriminator Networks (one from noise input, one from generated samples)
  discriminator = Discriminator()
  disc_real = discriminator(x)
  disc_fake = discriminator(gen_sample)

  # Build Loss
  y_ml = Classifier(gen_sample)
  gan_class_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_input, logits=y_ml))
  gen_loss = -tf.reduce_mean(tf.log(disc_fake)) + GAN_CLASS_COE*gan_class_loss
  disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

  # Build Optimizers
  optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
  optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

  # Training Variables for each optimizer
  # By default in TensorFlow, all variables are updated by each optimizer, so we
  # need to precise for each one of them the specific variables to update.
  # Generator Network Variables
  gen_vars = [Generator.linear_w1, Generator.linear_b1, Generator.linear_w2, Generator.linear_b2]
  # Discriminator Network Variables
  disc_vars = [Discriminator.linear_w1, Discriminator.linear_b1, Discriminator.linear_w2, Discriminator.linear_b2]

  # Create training operations
  train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
  train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)
  # Initialize the variables (i.e. assign their default value)
  init = tf.global_variables_initializer()

  # Start training
  with tf.Session() as sess:

      # Run the initializer
      sess.run(init)

      for i in range(1, num_steps+1):
          # Prepare Data
          # Get the next batch of MNIST data (only images are needed, not labels)
          batch_x, batch_y = mnist.train.next_batch(batch_size)
          # Generate noise to feed to the generator
          z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

          # Train
          feed_dict = {x: batch_x, label_input:batch_y, gen_input: z}
          _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                  feed_dict=feed_dict)
          if i % 1000 == 0 or i == 1:
              print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

      # Generate images from noise, using the generator network.
      f, a = plt.subplots(4, 10, figsize=(10, 4))
      for i in range(10):
          # Noise input.
          z = np.random.uniform(-1., 1., size=[4, noise_dim])
          g = sess.run([gen_sample], feed_dict={gen_input: z})
          g = np.reshape(g, newshape=(4, 28, 28, 1))
          # Reverse colours for better display
          g = -1 * (g - 1)
          for j in range(4):
              # Generate image from noise. Extend to 3 channels for matplot figure.
              img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                               newshape=(28, 28, 3))
              a[j][i].imshow(img)

      f.show()
      plt.draw()
      plt.waitforbuttonpress()
      
if __name__ == '__main__':
  train_gan()