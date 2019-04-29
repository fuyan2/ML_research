
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

tf.reset_default_graph()
tf.set_random_seed(1)

# Training Params
num_steps = 100000
learning_rate = 0.00002
x_dim = 2
noise_dim = 100
NUM_LABEL = 1
GAN_CLASS_COE = 0
gan_batch_size = 1000

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.), dtype=tf.float32)
 
# Linear Regression 
class Generator(object):
  # G Parameters
  def __init__(self, noise_dim, NUM_LABEL, batch_size):
    self.batch_size = batch_size
    self.linear_w1 = tf.Variable(glorot_init([noise_dim+NUM_LABEL, 100]),name='glw1')
    self.linear_b1 = tf.Variable(glorot_init([100]),name='glb1')
    self.linear_w2 = tf.Variable(glorot_init([100, 50]),name='glw2')
    self.linear_b2 = tf.Variable(glorot_init([50]),name='glb2')
    self.linear_w3 = tf.Variable(glorot_init([50, x_dim]),name='glw3')
    self.linear_b3 = tf.Variable(glorot_init([x_dim]),name='glb3')
    
    self.training = True

 # Build G Graph
  def __call__(self, z,y):
    z_y = tf.concat((z,y),1)
    linear_z1 = tf.nn.leaky_relu(tf.matmul(z_y,self.linear_w1) + self.linear_b1)
    linear_z2 = tf.nn.leaky_relu(tf.matmul(linear_z1,self.linear_w2) + self.linear_b2)
    out_layer = tf.matmul(linear_z2,self.linear_w3)+self.linear_b3
    return out_layer

# Logistic Regression
class Disciminator(object):
  # D Parameters
  def __init__(self):
    self.linear_w1 = tf.Variable(glorot_init([x_dim, 100]))
    self.linear_b1 = tf.Variable(glorot_init([100]))
    self.linear_w2 = tf.Variable(glorot_init([100, 1]))
    self.linear_b2 = tf.Variable(glorot_init([1]))

    self.training = True

  # Build D Graph
  def __call__(self, x):
    linear1 = tf.nn.leaky_relu(tf.matmul(x, self.linear_w1) + self.linear_b1)
    out = tf.matmul(linear1, self.linear_w2) + self.linear_b2
    out = tf.sigmoid(out)
    return out

class Classifier(object):
  def __init__(self):
    self.linear_w1 = tf.Variable(glorot_init([x_dim, 100]))
    self.linear_b1 = tf.Variable(glorot_init([100]))
    self.linear_w2 = tf.Variable(glorot_init([100, NUM_LABEL]))
    self.linear_b2 = tf.Variable(glorot_init([NUM_LABEL]))

    self.training = True

  # Build D Graph
  def __call__(self, x):
    linear1 = tf.nn.leaky_relu(tf.matmul(x, self.linear_w1) + self.linear_b1)
    out = tf.matmul(linear1, self.linear_w2) + self.linear_b2
    out = tf.sigmoid(out)
    return out
  
  
def plot_gan_image(name, epoch, sess):
  # Generate images from noise, using the generator network.
  fig, ax = plt.subplots()
  for i in range(2):
      # Desired label
      d_label = np.zeros([gan_batch_size, NUM_LABEL])
      d_label[:] = i
      # Noise input.
      z = np.random.uniform(-1., 1., size=[gan_batch_size, noise_dim])
      g = sess.run([gen_sample], feed_dict={gen_input: z, desired_label: d_label})
      g = np.reshape(g, [gan_batch_size,2])
      if i == 0:
        ax.scatter(g[:,0], g[:,1], c='b')
      if i == 1:
        ax.scatter(g[:,0], g[:,1], c='r')
     
  # f.show()
  plt.savefig('simple_gan_out/'+name+epoch)

# Build Dataset
features = tf.placeholder(tf.float32, shape=[None, x_dim])
labels = tf.placeholder(tf.float32, shape=[None, NUM_LABEL])
batch_size = tf.placeholder(tf.int64)
sample_size = tf.placeholder(tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
dataset = dataset.shuffle(sample_size, reshuffle_each_iteration=True)
dataset = dataset.batch(batch_size, drop_remainder=True).repeat()
iter = dataset.make_initializable_iterator()
next_batch = iter.get_next()

mu1 = [0., 0.]
sigma1 = [[1., 0.], [0., 1.]]
mu2 = [15., 0.]
sigma2 = [[1., 0.], [0., 1.]]
x_1 = np.random.multivariate_normal(mu1, sigma1, 10000)
x_2 = np.random.multivariate_normal(mu2, sigma2, 10000)
plt.scatter(x_1[:,0], x_1[:,1], c='r')
plt.scatter(x_2[:,0], x_2[:,1], c='b')
y_1 = np.zeros(10000)
y_2 = np.ones(10000)
x_train = np.concatenate((x_1, x_2), axis=0)
y_train = np.concatenate((y_1, y_2), axis=0)
y_train= y_train.reshape([20000, 1])
# train_data = np.concatenate((x_train, y_train), axis=1)
# print(train_data.shape, train_data)
# train_data = tf.random.shuffle(train_data)
# print(train_data.shape, train_data)
# x_train = train_data[:,0:1]
# y_train = train_data[:, 2]

#aux_data is same as data1
mu3 = [1., 1.]
mu4 = [6., 6.]

aux_data1 = np.random.multivariate_normal(mu1, sigma1, 500)
aux_data2 = np.random.multivariate_normal(mu2, sigma1, 500)
aux_data = np.concatenate((aux_data1, aux_data2), axis=0)
plt.scatter(aux_data[:,0], aux_data[:,1], c='y')
#plt.close()
plt.savefig("simple_gan_out/train_aux_data")
# Build The Classifier
x = tf.placeholder(tf.float32, shape=[None, x_dim])
y = tf.placeholder(tf.float32, shape=[None, NUM_LABEL])
model = Classifier()
y_ml = model(x)
model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_ml))

y_pred = tf.math.greater(y_ml, 0.5)
correct = tf.equal(tf.math.greater(y_ml, 0.5), tf.math.greater(y, 0.5))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

model_optimizer = tf.train.AdamOptimizer(0.001).minimize(model_loss, var_list=[model.linear_w1, model.linear_b1, model.linear_w2, model.linear_b2])

# Build Networks
# Network Inputs
gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
aux_x = tf.placeholder(tf.float32, shape=[None, x_dim])
desired_label = tf.placeholder(tf.float32, shape=[None, NUM_LABEL], name='desired_label')
# Build G Network
G = Generator(noise_dim, NUM_LABEL, gan_batch_size)
gen_sample = G(gen_input,desired_label)

# Build 2 D Networks (one from noise input, one from generated samples)
D = Disciminator()
disc_real = D(aux_x)
disc_fake = D(gen_sample)

# Build Loss
#   y_ml = model(gen_sample)
gan_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=desired_label, logits=y_ml))
gen_loss = -tf.reduce_mean(tf.log(disc_fake)) + GAN_CLASS_COE*gan_class_loss
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# G Network Variables
gen_vars = [G.linear_w1, G.linear_b1, G.linear_w2, G.linear_b2]
# D Network Variables
disc_vars = [D.linear_w1, D.linear_b1, D.linear_w2, D.linear_b2]

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

def train_gan():
  # mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
  # Train Classifier
  
  # Initialize the variables (i.e. assign their default value)
  init = tf.global_variables_initializer()

  # Start training
  with tf.Session() as sess:

      # Run the initializer
      sess.run(init)

      sess.run(iter.initializer, feed_dict = {features: x_train, labels: y_train, batch_size: gan_batch_size, sample_size: 20000})

      # Train the Classifier First
      for i in range(2000):
        batch = sess.run(next_batch)
        model_optimizer.run(feed_dict={ x: batch[0], y: batch[1]})
        if i % 200 == 0:
          train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1] })
          print('Epoch %d, training accuracy %g' % (i, train_accuracy))    

      test_acc, y_prediction = sess.run([accuracy, y_pred], feed_dict={x: x_train, y: y_train})
      print("test acc:", test_acc)

      cluster1 = []
      cluster2 = []
      for i in range(0, 20000):
        if y_prediction[i]:
          cluster2.append(x_train[i])
        else:
          cluster1.append(x_train[i])

      cluster1 = np.array(cluster1)
      cluster2 = np.array(cluster2)
      print('cluster1: ', cluster1.shape, 'cluster2: ', cluster2.shape)
      plt.scatter(cluster1[:,0], cluster1[:,1],c='r')
      plt.scatter(cluster2[:,0], cluster2[:,1],c='b')
      plt.savefig('simple_gan_out/Classifier_plot')
      
      #Train GAN
      for i in range(100000):
        batch_x = aux_data
        # Use one hot aux_label instead of distribution
        aux_label = sess.run(y_ml, feed_dict={x: batch_x})
        #Use one hot label
#         aux_label = np.select([aux_label > 0.5, aux_label <= 0.5], [np.ones(100), np.zeros(100)])
#         aux_label = np.zeros([100,1])
        # Sample random noise 
        z = np.random.uniform(-1., 1., size=[gan_batch_size, noise_dim])
        # Pass random noise into the generator and get generated image
        gen_mi = sess.run(gen_sample, feed_dict={gen_input:z, desired_label: aux_label}) 
       
        #! Train Discriminator
        train_disc.run(feed_dict={aux_x: batch_x,  gen_input: z, x: gen_mi, desired_label: aux_label})
        if i % 5 == 0:
          train_gen.run(feed_dict={aux_x: batch_x,  gen_input: z, x: gen_mi, desired_label: aux_label})
       
        if i % 2000 == 0:
          gl,dl = sess.run([gen_loss, disc_loss], feed_dict={aux_x: batch_x,  gen_input: z, x: gen_mi, desired_label: aux_label})
          print('Epoch %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
          plot_gan_image('gan_out',str(i), sess)
      
      
if __name__ == '__main__':
  train_gan()