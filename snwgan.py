
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tf_SNDCGAN.libs.sn import spectral_normed_weight

# Training Params
num_steps = 100000
learning_rate = 0.0002

# Network Params
# noise_dim = 100 # Noise data points
# NUM_LABEL = 10 #10 labels

SPECTRAL_NORM_UPDATE_OPS = "spectral_norm_update_ops"

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))
  
class Generator:
  # Generator Parameters
  def __init__(self, noise_dim, NUM_LABEL, batch_size):
    self.batch_size = batch_size
    self.linear_w = tf.Variable(glorot_init([noise_dim+NUM_LABEL, 7*7*256]),name='glw')
    self.linear_b = tf.Variable(glorot_init([7*7*256]),name='glb')

    self.deconv_w1 = tf.Variable(glorot_init([4, 4, 128, 256]),name='gdw1')
    self.deconv_w2 = tf.Variable(glorot_init([4, 4, 64, 128]),name='gdw2')
    self.deconv_w3 = tf.Variable(glorot_init([3, 3, 1, 64]),name='gdw3')

    self.deconv_b1 = tf.Variable(tf.constant(0.1, shape=[128]), name='gb1')
    self.deconv_b2 = tf.Variable(tf.constant(0.1, shape=[64]), name='gb2')
    self.deconv_b3 = tf.Variable(tf.constant(0.1, shape=[1]), name='gb3')

    self.training = True

 # Build Generator Graph
  def __call__(self, z,y):
    z_y = tf.concat([z,y],1)
    linear_h = tf.matmul(z_y,self.linear_w)+self.linear_b
    linear_h_reshape = tf.reshape(linear_h , [-1,7, 7,256])  
    deconv_xw1 = tf.nn.conv2d_transpose(linear_h_reshape, self.deconv_w1,output_shape=[self.batch_size,14,14,128], strides=[1, 2, 2, 1])
    xw1_norm = tf.layers.batch_normalization(deconv_xw1, training=self.training)
    deconv_h1 = tf.nn.leaky_relu(xw1_norm + self.deconv_b1)
    deconv_xw2 = tf.nn.conv2d_transpose(deconv_h1, self.deconv_w2,output_shape=[self.batch_size,28,28,64], strides=[1, 2, 2, 1])
    xw2_norm = tf.layers.batch_normalization(deconv_xw2, training=self.training)
    deconv_h2 = tf.nn.leaky_relu(xw2_norm + self.deconv_b2)
    deconv_xw3 = tf.nn.conv2d_transpose(deconv_h2, self.deconv_w3,output_shape=[self.batch_size,28,28,1], strides=[1, 1, 1, 1])
    out_layer = tf.nn.sigmoid(deconv_xw3)
    # out_layer = tf.nn.tanh(deconv_xw3)
    return out_layer

class Discriminator:
  # Discriminator Parameters
  def __init__(self):
    self.conv_w1 = tf.get_variable('dw1', shape=[3, 3, 1, 64])
    self.conv_w2 = tf.get_variable('dw2', shape=[4, 4, 64, 128])
    self.conv_w3 = tf.get_variable('dw3', shape=[3, 3, 128, 128])
    self.conv_w4 = tf.get_variable('dw4', shape=[4, 4, 128, 256])
    self.conv_w5 = tf.get_variable('dw5', shape=[3, 3, 256, 256])

    self.conv_b1 = tf.Variable(tf.constant(0.1, shape=[64]), name='db1') 
    self.conv_b2 = tf.Variable(tf.constant(0.1, shape=[128]), name='db2') 
    self.conv_b3 = tf.Variable(tf.constant(0.1, shape=[128]), name='db3') 
    self.conv_b4 = tf.Variable(tf.constant(0.1, shape=[256]), name='db4') 
    self.conv_b5 = tf.Variable(tf.constant(0.1, shape=[256]), name='db5') 

    self.linear_w1 = tf.Variable(glorot_init([7*7*256, 300]))
    self.linear_b1 = tf.Variable(glorot_init([300]))
    self.linear_w2 = tf.Variable(glorot_init([300, 1]))
    self.linear_b2 = tf.Variable(glorot_init([1]))

    self.training = True

  # Build Discriminator Graph
  def __call__(self, x):
    conv_xw1 = tf.nn.conv2d(x,spectral_normed_weight(self.conv_w1, num_iters=1, update_collection=SPECTRAL_NORM_UPDATE_OPS),strides=[1, 1, 1, 1], padding='SAME')
    xw1_norm = tf.layers.batch_normalization(conv_xw1, training=self.training)
    conv_h1 = tf.nn.leaky_relu(xw1_norm+ self.conv_b1)
    conv_xw2 = tf.nn.conv2d(conv_h1,spectral_normed_weight(self.conv_w2, num_iters=1, update_collection=SPECTRAL_NORM_UPDATE_OPS),strides=[1, 2, 2, 1], padding='SAME')
    xw2_norm = tf.layers.batch_normalization(conv_xw2, training=self.training)
    conv_h2 = tf.nn.leaky_relu(xw2_norm+ self.conv_b2)
    conv_xw3 = tf.nn.conv2d(conv_h2,spectral_normed_weight(self.conv_w3, num_iters=1, update_collection=SPECTRAL_NORM_UPDATE_OPS),strides=[1, 1, 1, 1], padding='SAME')
    xw3_norm = tf.layers.batch_normalization(conv_xw3, training=self.training)
    conv_h3 = tf.nn.leaky_relu(xw3_norm+ self.conv_b3)
    conv_xw4 = tf.nn.conv2d(conv_h3,spectral_normed_weight(self.conv_w4, num_iters=1, update_collection=SPECTRAL_NORM_UPDATE_OPS),strides=[1, 2, 2, 1], padding='SAME')
    xw4_norm = tf.layers.batch_normalization(conv_xw4, training=self.training)
    conv_h4 = tf.nn.leaky_relu(xw4_norm+ self.conv_b4)
    conv_xw5 = tf.nn.conv2d(conv_h4,spectral_normed_weight(self.conv_w5, num_iters=1, update_collection=SPECTRAL_NORM_UPDATE_OPS),strides=[1, 1, 1, 1], padding='SAME')
    xw5_norm = tf.layers.batch_normalization(conv_xw5, training=self.training)
    conv_h5 = tf.nn.leaky_relu(xw5_norm+ self.conv_b5)
    conv_h5_flat = tf.reshape(conv_h5, [-1, 7*7*256])
    linear1 = tf.matmul(conv_h5_flat, spectral_normed_weight(self.linear_w1)) + self.linear_b1
    linear1 = tf.nn.leaky_relu(linear1)

    out = tf.matmul(linear1, spectral_normed_weight(self.linear_w2)) + self.linear_b2
    #! out = tf.sigmoid(out)
    return out



def train_gan():
  mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
  tf.reset_default_graph()
  tf.set_random_seed(1)
  # Build Networks
  # Network Inputs
  gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
  x = tf.placeholder(tf.float32, shape=[None, 28*28])
  disc_input = tf.reshape(x , [-1,28, 28,1])
  label_input = tf.placeholder(tf.float32, shape=[None, NUM_LABEL], name='label_input')
  # Build Generator Network
  generator = Generator()
  gen_sample = generator.build(gen_input,label_input)

  # Build 2 Discriminator Networks (one from noise input, one from generated samples)
  discriminator = Discriminator()
  disc_real = discriminator.build(disc_input)
  disc_fake = discriminator.build(gen_sample)

  # Build Loss
  gen_loss = -tf.reduce_mean(tf.log(disc_fake))
  disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

  # Build Optimizers
  optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
  optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

  # Training Variables for each optimizer
  # By default in TensorFlow, all variables are updated by each optimizer, so we
  # need to precise for each one of them the specific variables to update.
  # Generator Network Variables
  gen_vars = [Generator.linear_w, Generator.linear_b, Generator.deconv_w1, Generator.deconv_w2, Generator.deconv_w3, 
              Generator.deconv_b1, Generator.deconv_b2, Generator.deconv_b3]
  # Discriminator Network Variables
  disc_vars = [Discriminator.conv_w1, Discriminator.conv_w2, Discriminator.conv_w3, Discriminator.conv_w4, Discriminator.conv_w5,
              Discriminator.conv_b1, Discriminator.conv_b2, Discriminator.conv_b3, Discriminator.conv_b4, Discriminator.conv_b5,
              Discriminator.linear_w, Discriminator.linear_b]

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