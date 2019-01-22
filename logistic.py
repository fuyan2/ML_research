import time
import numpy as np
import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pdb
import copy
import multiprocessing as mp

from datetime import datetime
from skimage.measure import compare_ssim

#Defining Parameters
IMG_ROW = 28
IMG_COL = 28
NUM_LABEL = 10
INV_HIDDEN = 5000
EPOCHS = 100
learning_rate = 0.1
loss_beta = 0.003
BATCH_SIZE =250

#Universal Setup Construct Dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, [x_train.shape[0], IMG_ROW*IMG_COL])
x_test = np.reshape(x_test, [x_test.shape[0], IMG_ROW*IMG_COL])
y_train = np.reshape(y_train, [y_train.shape[0], -1])
y_test = np.reshape(y_test, [y_test.shape[0], -1])

features = tf.placeholder(tf.float32, shape=[None, IMG_ROW * IMG_COL])
labels = tf.placeholder(tf.int32, shape=[None, 1])
batch_size = tf.placeholder(tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)
iter = dataset.make_initializable_iterator()
x, y_ = iter.get_next()
y = tf.one_hot(tf.reshape(y_,[-1]), NUM_LABEL)

print(x)
print(y)
print(y_train)
print(y_train.shape)

xavier_initializer = tf.contrib.layers.xavier_initializer()

def layer(input, num_units):
  W = tf.Variable(tf.zeros([input.shape[1], num_units]), name="w")
  B = tf.Variable(tf.zeros([num_units]), name="b")
  output = tf.matmul(input,W) + B
  return W, B, output

def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def inverter(y, w, b):
  with tf.variable_scope('InvLayer_1', reuse=tf.AUTO_REUSE) as scope:
    wx = tf.add_n([
      tf.matmul(tf.reshape(w, [1, -1]), inv_weights['inv_w']),
      tf.matmul(tf.reshape(b, [1, -1]), inv_weights['inv_b']),
    ])
    wy = tf.matmul(y, inv_weights['h_label'])

    wt = tf.add(wy, wx)

    hidden_layer =  tf.add(wt, inv_biases['h_b'])
    rect = lrelu(hidden_layer, 0.3)

  # Layer 2, rectifier with output IMG_ROWS * IMG_COLS
  with tf.variable_scope('DLayer_2') as scope:
    out_layer = tf.add(tf.matmul(rect, inv_weights['inv_out']), inv_biases['inv_out'])
    rect = lrelu(out_layer, 0.3)
  return rect

#Build Logistic Layer
with tf.name_scope("logistic_layer"):
  w,b,z = layer(x,NUM_LABEL)
  y_ml = tf.nn.softmax(z)

#Build Inverter Regularizer
model_weights = tf.concat([tf.reshape(w,[1, -1]),tf.reshape(b,[1, -1])], 1)
inv_weights = {
  'inv_w': tf.Variable(tf.zeros([tf.reshape(w, [-1]).shape[0], INV_HIDDEN])),
  'inv_b': tf.Variable(tf.zeros([tf.reshape(b, [-1]).shape[0], INV_HIDDEN])),
  'h_label': tf.Variable(tf.zeros([NUM_LABEL, INV_HIDDEN])),
  'inv_out': tf.Variable(tf.zeros([INV_HIDDEN, IMG_ROW * IMG_COL]))
}
inv_biases = {
  'h_b': tf.Variable(tf.zeros([INV_HIDDEN])),
  'inv_out': tf.Variable(tf.zeros([IMG_ROW * IMG_COL])),
}

inv_x = inverter(y, w, b)

#Calculate loss
class_loss = tf.losses.softmax_cross_entropy(y,y_ml)
inv_loss = tf.losses.mean_squared_error(labels=x, predictions=inv_x)

# calculate prediction accuracy
correct = tf.equal(tf.argmax(y_ml, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

def train(loss_beta, learning_rate, Epoch):
  total_loss = class_loss - loss_beta * inv_loss
  model_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
  inverter_optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(inv_loss)
  init_vars = tf.global_variables_initializer()
  
  with tf.Session() as sess:
    sess.run(init_vars)
    # initialise iterator with train data
    sess.run(iter.initializer, feed_dict = {features: x_train, labels: y_train, batch_size: BATCH_SIZE})
    
    print('Training...')
    for i in range(Epoch):
      sess.run(model_optimizer)
      sess.run(inverter_optimizer)
      train_acc = sess.run(accuracy)
      print("step %g train accuracy is %g"%(i, train_acc))
      
    # initialise iterator with test data
    sess.run(iter.initializer, feed_dict = {features: x_test, labels: y_test, batch_size: y_test.shape[0]})
    test_acc = sess.run(accuracy)
    print('Test accuracy: {:4f}'.format(test_acc))
      
    return test_acc

betas = [0, 0.001, 0.01, 0.1, 0.5, 1., 2., 5., 7., 10., 15., 20.]
test_accs = np.zeros(len(betas))
for i,beta in enumerate(betas):
  test_accs[i] = train(betas,0.01,500)
  print("beta is %g, test accuracy is %g"%(beta, test_accs[i]))
  
plt.plot(betas, test_acc)