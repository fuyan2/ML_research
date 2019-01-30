import time
import numpy as np
import sys
import os
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb
import copy
import multiprocessing as mp

from datetime import datetime
from skimage.measure import compare_ssim
from tensorflow.examples.tutorials.mnist import input_data

#Defining Initial Parameters
IMG_ROWS = 28
IMG_COLS = 28
NUM_LABEL = 10
INV_HIDDEN = 5000
EPOCHS = 100
learning_rate = 0.1
loss_beta = 0.003
BATCH_SIZE = 250

#Flatten input dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, [x_train.shape[0], -1])
x_test = np.reshape(x_test, [x_test.shape[0], -1])
y_train = np.reshape(y_train, [y_train.shape[0], -1])
y_test = np.reshape(y_test, [y_test.shape[0], -1])

#construct dataset
features = tf.placeholder(tf.float32, shape=[None, IMG_ROWS * IMG_COLS])
labels = tf.placeholder(tf.int32, shape=[None, 1])
batch_size = tf.placeholder(tf.int64)
sample_size = tf.placeholder(tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
dataset = dataset.shuffle(sample_size, seed=1, reshuffle_each_iteration=True)
dataset = dataset.batch(batch_size).repeat()

iter = dataset.make_initializable_iterator()
x, y_ = iter.get_next()
y = tf.one_hot(tf.reshape(y_,[-1]), NUM_LABEL)

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# x = tf.placeholder(tf.float32, shape=[None, IMG_ROWS * IMG_COLS])
# y = tf.placeholder(tf.float32, shape=[None, 10])

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

def train(loss_beta, learning_rate, Epoch, batch):
  total_loss = class_loss - loss_beta * mi_loss
  model_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, var_list=[w,b])
  inverter_optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(inv_loss)
  init_vars = tf.global_variables_initializer()
  
  with tf.Session() as sess:
    sess.run(init_vars)
   
    # initialise iterator with train data
    sess.run(iter.initializer, feed_dict = {features: x_train, labels: y_train, batch_size: batch, sample_size: y_train.shape[0]})
    
    print('Beta %g Training...'%(loss_beta))
    for i in range(Epoch):
      # batch = mnist.train.next_batch(Batch)
      # model_optimizer.run(feed_dict={ x: batch[0], y: batch[1]})
      # inverter_optimizer.run(feed_dict={ x: batch[0], y: batch[1]})
      _,_,train_acc,train_total_loss, train_inv_loss, train_class_loss = sess.run([model_optimizer,inverter_optimizer, accuracy, total_loss, inv_loss, class_loss])
      if i % 1000 == 0:  
        print("step %g train accuracy is %g, total_loss is %g, inv_loss is %g, class_loss is %g"%(i, train_acc,train_total_loss, train_inv_loss, train_class_loss))
        # train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1] })
        # valid_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, y: mnist.validation.labels })
        # print('step %d, training accuracy %g, validation accuracy %g' % (i, train_accuracy,valid_accuracy))
      
    # initialise iterator with test data
    sess.run(iter.initializer, feed_dict = {features: x_test, labels: y_test, batch_size: y_test.shape[0], sample_size: 1})
    test_acc = sess.run(accuracy, feed_dict={ x: mnist.test.images, y: mnist.test.labels })
    print("beta is %g, test accuracy is %g"%(loss_beta, test_acc))
      
    return test_acc

#Will not run when file is imported by other files
if __name__ == '__main__':
  betas = [0.0, 1., 10., 15., 20.]

  test_accs = np.zeros(len(betas))

  # Iterate through beta
  for i,beta in enumerate(betas):
    test_accs[i] = train(beta,0.1, 20000, 250)

  # Iterate through batch size
  # for i,batch in enumerate(batchs):
  #   test_accs[i] = train(0,0.01,500, batch)

  # for i,rate in enumerate(learn_rate):
  #   test_accs[i] = train(0,rate,20000, 250)

  np.save("logreg_acc_rate0", test_accs)

  plt.plot(betas, test_accs)
  plt.show()