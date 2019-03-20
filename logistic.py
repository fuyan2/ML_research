import time
import numpy as np
import sys
import os
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import pdb
import copy
import multiprocessing as mp

from datetime import datetime
from skimage.measure import compare_ssim
from tensorflow.examples.tutorials.mnist import input_data
import math
inf = math.inf

tf.reset_default_graph()
tf.set_random_seed(1)
#Defining Initial Parameters
IMG_ROWS = 28
IMG_COLS = 28
NUM_LABEL = 10
INV_HIDDEN = 5000
EPOCHS = 100
learning_rate = 0.1
loss_beta = 0.003
BATCH_SIZE = 250
ALPHA = 20000
BETA = 5
GAMMA = 1
LAMBDA = 0.06

#Flatten input dataset
# mnist = tf.keras.datasets.mnist
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train = np.reshape(x_train, [x_train.shape[0], -1])
# x_test = np.reshape(x_test, [x_test.shape[0], -1])
# y_train = np.reshape(y_train, [y_train.shape[0], -1])
# y_test = np.reshape(y_test, [y_test.shape[0], -1])

# #construct dataset
# features = tf.placeholder(tf.float32, shape=[None, IMG_ROWS * IMG_COLS])
# labels = tf.placeholder(tf.int32, shape=[None, 1])
# batch_size = tf.placeholder(tf.int64)
# sample_size = tf.placeholder(tf.int64)
# dataset = tf.data.Dataset.from_tensor_slices((features, labels))
# dataset = dataset.shuffle(sample_size, reshuffle_each_iteration=True)
# dataset = dataset.batch(batch_size).repeat()

# iter = dataset.make_initializable_iterator()
# x, y_ = iter.get_next()
# y = tf.one_hot(tf.reshape(y_,[-1]), NUM_LABEL)

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

def train(loss_beta, learning_rate, Epoch, Batch, LABEL):
  # total_loss = class_loss - loss_beta * mi_loss
  total_loss = class_loss + loss_beta * tf.norm(model_weights,ord=1)
  model_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, var_list=[w,b])
  # inverter_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(inv_loss)
  init_vars = tf.global_variables_initializer()
  
  with tf.Session() as sess:
    sess.run(init_vars)
   
    # initialise iterator with train data
    # sess.run(iter.initializer, feed_dict = {features: x_train, labels: y_train, batch_size: Batch, sample_size: 10000})
    
    print('Beta %g Training...'%(loss_beta))
    for i in range(Epoch):
      batch = mnist.train.next_batch(Batch)
      model_optimizer.run(feed_dict={ x: batch[0], y: batch[1]})
      # inverter_optimizer.run(feed_dict={ x: batch[0], y: batch[1]})
      # _,_,train_acc,train_total_loss, train_inv_loss, train_class_loss = sess.run([model_optimizer,inverter_optimizer, accuracy, total_loss, inv_loss, class_loss])
      if i % 1000 == 0:  
        # print("step %g train accuracy is %g, total_loss is %g, inv_loss is %g, class_loss is %g"%(i, train_acc,train_total_loss, train_inv_loss, train_class_loss))
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1] })
        valid_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, y: mnist.validation.labels })
        print('step %d, training accuracy %g, validation accuracy %g' % (i, train_accuracy,valid_accuracy))    
    test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels })

    # initialise iterator with test data
    # sess.run(iter.initializer, feed_dict = {features: x_test, labels: y_test, batch_size: y_test.shape[0], sample_size: 1})
    # test_acc = sess.run(accuracy)
    print("beta is %g, test accuracy is %g"%(loss_beta, test_acc))
    
    # MODEL INVERSION
    if(LABEL == None):
      inverted_images = np.zeros((NUM_LABEL, IMG_ROWS*IMG_COLS))
      for i in range(NUM_LABEL):
        print("i = %d" % i)
        inverted_images[i] = model_inversion(i, y_ml, sess, 1)[0]
    else:
      inverted_images = model_inversion(LABEL, y_ml, sess, 1)[0]

      
    return test_acc, inverted_images
    
def model_inversion(i, y_conv, sess, iterate):
  label_chosen = np.zeros(10)
  label_chosen[i] = 1
  
  cost_x = 1 - tf.squeeze(tf.gather(y_conv, i, axis=1), 0)
  gradient_of_cost = tf.gradients(cost_x, x)
  x_inv = x - tf.scalar_mul(LAMBDA, tf.squeeze(gradient_of_cost, 0))
  x_mi = np.zeros((1, 784))
  previous_costs = [inf,inf,inf,inf,inf]

  for i in range(ALPHA):
    x_mi = sess.run(x_inv, feed_dict={x: x_mi, y: [label_chosen] })
    cost_x_mi = sess.run(cost_x, feed_dict={x: x_mi, y: [label_chosen] })
    max_cost = max(previous_costs)
    
    if(cost_x_mi > max_cost or (iterate and cost_x_mi == 0)):
      print("Early break, no ALPHA HIT")
      break;
    else:
      previous_costs.append(cost_x_mi)
      previous_costs.pop(0)

    if(i % 1000 == 0):
      print('step %d, current cost is %g' % (i, cost_x_mi))

  print('iteration hit:', i+1)

  # Make background black instead of grey
  for i in range(x_mi.shape[1]):
    if(x_mi[0][i] < 0):
      x_mi[0][i] = 0
    if(x_mi[0][i] > 1):
      x_mi[0][i] = 1

  check_pred = sess.run(correct, feed_dict={x: x_mi, y: [label_chosen] })
  print("Prediction for reconstructed image:", check_pred)
  return x_mi

def show_image(array):
  adv_img = plt.imshow(np.reshape(array, (28, 28)), cmap="gray", vmin=array.min(), vmax=array.max())
  plt.show(adv_img)

def average_images(images, labels):
  avg_imgs = np.zeros((10, IMG_COLS * IMG_ROWS))

  for i in range(10):
    indices = np.where(labels == i)
    imgs_for_label = images[indices]
    avg_imgs[i] = 1/imgs_for_label.shape[0] * np.sum(imgs_for_label, axis=0)

  return avg_imgs 

def square_mean(img1, img2):
  img1 = np.reshape(img1, -1)
  img2 = np.reshape(img2, -1)
  diff = img1 - img2
  diff_sq = np.square(diff)
  multiplier = 1 / (IMG_ROWS*IMG_COLS)
  return multiplier * np.sum(diff_sq)

def show_all_images_one_row(array):
    images = []
    for i in range(array.shape[0]):
      images.append(np.reshape(array[i], (IMG_ROWS, IMG_COLS)))

    all_concat = np.concatenate(images, 1)
    plt.imshow(all_concat, cmap="gray")
    plt.show()

def plot_vs_beta(values, betas, title, axis):
  plt.plot(betas, values, 'ko-')
  plt.ylabel(axis)
  plt.title(title)
  plt.xlabel('Beta')
  plt.xscale('log')
  plt.show()

def measure_over_beta(LABEL):
  betas = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
  mnist_noh = input_data.read_data_sets('MNIST_data', one_hot=False)
  average_image = average_images(mnist_noh.train.images, mnist_noh.train.labels)[LABEL]
  test_accs = np.zeros(len(betas))
  ssim = np.zeros(len(betas))
  sq_mean = np.zeros(len(betas))
  image_over_beta = np.zeros((len(betas), IMG_ROWS*IMG_COLS))
  # Iterate through beta
  for i,beta in enumerate(betas):
    test_accs[i], inverted_image = train(beta, 0.1, 10000, 250, LABEL)
    norm_mi = (inverted_image + inverted_image.min()) * 1 /inverted_image.max()
    image_over_beta[i] = norm_mi
    ssim[i] = compare_ssim(np.reshape(average_image, (28, 28)), np.reshape(norm_mi, (28, 28)), data_range=1.0 - 0.0)  
    sq_mean[i] = square_mean(average_image, norm_mi)
  
  show_all_images_one_row(image_over_beta)
  plot_vs_beta(test_accs, betas, "Test Accuracy vs. Beta", "Accuracy (%)")
  plot_vs_beta(ssim, betas, "SSIM vs. Beta", "SSIM")
  plot_vs_beta(sq_mean, betas, "Square Mean over Pixels vs. Beta", "Square Mean")



#Will not run when file is imported by other files
if __name__ == '__main__':
  measure_over_beta(0)
#   betas = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

#   test_accs = np.zeros(len(betas))
#   acc, inverted_images = train(1, 0.01, 200, 100)
#   for i in range(10):
#     show_image(inverted_images[i])
#   # Iterate through beta
# #   for i,beta in enumerate(betas):
# #     test_accs[i] = train(beta,0.01, 200, 250)

#   np.save("logreg_acc_rate0", test_accs)

#   plt.plot(betas, test_accs)
#   plt.show()