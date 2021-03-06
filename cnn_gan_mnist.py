import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image
import os
import gzip
import struct
import array
import sys

import numpy as np
import tensorflow as tf
import random
import math
# from skimage.measure import compare_ssim
from tf_models import *
from utils import *
# from snwgan import snw_Generator, snw_Discriminator, cnn_Discriminator
inf = 1e9


# Training Params
num_steps = 100000
learning_rate = 0.0002 #0.00002
x_dim = 28*28
noise_dim = 128 #20
IMG_ROWS = 28
IMG_COLS = 28
NUM_LABEL = 10 #10
GAN_CLASS_COE = 100 #10
gan_batch_size = 250
num_data = 10000
INV_HIDDEN = 100 #5000
beta = 0 #1, 0.5
model_l2 = 0 #0.0001 
wasserstein = True
cnn_gan = True
inverter_take_avgimg = False
gan_epoch = 100
class_epoch = 63
#Fredrickson Params
ALPHA = 20000
BETA = 5
GAMMA = 1
LAMBDA = 0.06

one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)

###################### Build Dataset #############################
features = tf.placeholder(tf.float32, shape=[None, x_dim])
labels = tf.placeholder(tf.float32, shape=[None, NUM_LABEL])
batch_size = tf.placeholder(tf.int64)
sample_size = tf.placeholder(tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
dataset = dataset.shuffle(sample_size, reshuffle_each_iteration=True)
dataset = dataset.batch(batch_size, drop_remainder=True).repeat()
iterator = dataset.make_initializable_iterator()
next_batch = iterator.get_next()

#Loading data
digits_size, digits_x_train, digits_y_train, digits_x_test, digits_y_test = load_mnist('digits')
letters_size, letters_x_train, letters_y_train, letters_x_test, letters_y_test = load_mnist('letters')

reduce_train = False
sep_point = 60000 # 50%, 25%
if reduce_train:
    digits_y_train = digits_y_train[:sep_point]
    digits_x_train = digits_x_train[:sep_point, :]

# print('training dataset size:', digits_size)
avg_imgs = average_images(NUM_LABEL, x_dim, digits_x_train, digits_y_train)
avg_imgs = np.where(avg_imgs<0,0, avg_imgs)
avg_imgs = np.where(avg_imgs>1, 1, avg_imgs)

avg_digit_img = np.mean(digits_x_train, axis=0)

print('train data size is ', digits_x_train.shape[0])
print('test data size is ', digits_x_test.shape[0])
# print("aux data size is ", aux_x_data.shape[0])
y_train_one_hot = one_hot(digits_y_train, 10)
y_test_one_hot = one_hot(digits_y_test, 10)

################### Build The Classifier ####################
x = tf.placeholder(tf.float32, shape=[None, x_dim])
x_unflat = tf.reshape(x , [-1,IMG_ROWS, IMG_ROWS,1])
y = tf.placeholder(tf.float32, shape=[None, NUM_LABEL])
model = CNN_Classifier(NUM_LABEL, x_dim)
y_ml = model(x_unflat)
model_var = [model.conv_w1, model.conv_w2, model.conv_b1, model.conv_b2, model.full_w, model.full_b, model.out_w, model.out_b]
#Build Inverter Regularizer
# model_weights = tf.concat([tf.reshape(model.out_w,[1, -1]),tf.reshape(model.out_b,[1, -1])], 1)
# weight_shape = int(model_weights.shape[1])
# inverter = Inverter_Regularizer(NUM_LABEL, x_dim, weight_shape, INV_HIDDEN)
# avg_digit_img_inv = tf.constant(avg_digit_img, dtype=tf.float32)
# avg_digit_img_inv_reshape = tf.reshape(avg_digit_img_inv,[1,x_dim])

# if inverter_take_avgimg:
#     inv_x = inverter(y, model_weights, avg_digit_img_inv_reshape)
# else:
#     inv_x = inverter(y, model_weights, None)

# Calculate MODEL Loss
# inv_loss = tf.losses.mean_squared_error(labels=x, predictions=inv_x)
class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_ml))
# model_loss = class_loss - beta * inv_loss

correct = tf.equal(tf.argmax(y_ml, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Build Optimizer !Use model_loss
learning_rate = tf.placeholder(tf.float32, name='lr')
model_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(class_loss, var_list=model_var)
# inverter_optimizer = tf.train.AdamOptimizer(0.001).minimize(inv_loss, var_list=[inverter.w_model, inverter.w_label, inverter.w_out, inverter.b_in, inverter.b_out])

#################### Build GAN Networks ############################
# Network Inputs
gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
aux_x = tf.placeholder(tf.float32, shape=[None, x_dim])
aux_x_unflat = tf.reshape(aux_x, [gan_batch_size, 28, 28, 1])
aux_label = model(aux_x_unflat)
desired_label = tf.placeholder(tf.float32, shape=[None, NUM_LABEL])


# Build G Networks
# Use CNN gan
if cnn_gan:
    G = cnn_Generator(noise_dim, NUM_LABEL, gan_batch_size)
    gen_sample_unflat = G(gen_input,desired_label)
    gen_sample = tf.reshape(gen_sample_unflat,[gan_batch_size, x_dim])
    gen_vars = [G.linear_w, G.linear_b, G.deconv_w1, G.deconv_w2, G.deconv_w3]
    gen_weights = tf.concat([tf.reshape(G.linear_w, [1, -1]), tf.reshape(G.linear_b, [1, -1]), tf.reshape(G.deconv_w1, [1, -1]), tf.reshape(G.deconv_w2, [1, -1]), tf.reshape(G.deconv_w3, [1, -1])], 1)
    gen_label = model(gen_sample_unflat)

    # Build 2 D Networks (one from noise input, one from generated samples)
    D = cnn_Discriminator(NUM_LABEL) 
    disc_real = D(aux_x_unflat, aux_label)
    disc_fake = D(gen_sample_unflat, desired_label)

    # D Network Variables
    disc_vars = [D.linear_w1, D.linear_b1, D.linear_w2, D.conv_w1, D.conv_w2, D.conv_w3, D.conv_w4]

else:
    G = Generator(noise_dim, NUM_LABEL, x_dim, gan_batch_size)
    gen_sample = G(gen_input,desired_label)
    gen_sample_unflat = tf.reshape(gen_sample, [gan_batch_size, IMG_ROWS, IMG_COLS, 1])
    # G Network Variables
    gen_vars = [G.linear_w1, G.linear_b1, G.linear_w2, G.linear_b2, G.linear_w3, G.linear_b3]
    gen_weights = tf.concat([tf.reshape(G.linear_w1,[1, -1]), tf.reshape(G.linear_b1,[1, -1]), tf.reshape(G.linear_w2,[1, -1]), tf.reshape(G.linear_b2,[1, -1]), tf.reshape(G.linear_w3,[1, -1]), tf.reshape(G.linear_b3,[1, -1])], 1)
    gen_label = model(gen_sample_unflat)

    # Build 2 D Networks (one from noise input, one from generated samples)
    D = Disciminator(NUM_LABEL, x_dim, wasserstein) 
    disc_real = D(aux_x, aux_label)
    disc_fake = D(gen_sample, gen_label)

    # D Network Variables
    disc_vars = [D.linear_w1, D.linear_b1, D.linear_w2, D.linear_b2]

# Build Loss
similarity = tf.reduce_sum(tf.multiply(aux_label, desired_label), 1, keepdims=True )
gan_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=desired_label, logits=gen_label))

if wasserstein:
    gen_loss = -tf.reduce_mean(disc_fake) + GAN_CLASS_COE*gan_class_loss #+ 1. * tf.nn.l2_loss(gen_weights) #0.007, only need when no auxiliary
    disc_loss = -tf.reduce_mean(similarity*(disc_real - disc_fake)) #+ wgan_grad_pen(gan_batch_size,aux_x, aux_label, gen_sample)
    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in disc_vars] #0.01!
    train_gen = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(gen_loss, var_list=gen_vars)
    train_disc = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(disc_loss, var_list=disc_vars)
else:  
    gen_loss = -tf.reduce_mean(tf.log(tf.maximum(0.00001, disc_fake))) + GAN_CLASS_COE*gan_class_loss + 0.01 * tf.nn.l2_loss(gen_weights) #0.007, only need when no auxiliary
    disc_loss = -tf.reduce_mean(similarity*(tf.log(tf.maximum(0.0000001, disc_real)) + tf.log(tf.maximum(0.0000001, 1. - disc_fake)))) 
    train_gen = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(gen_loss, var_list=gen_vars)
    train_disc = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(disc_loss, var_list=disc_vars)
    
# Fredrickson Model Inversion Attack, Gradient Attack
def fred_mi(i, y_conv, sess, iterate):
    label_chosen = np.zeros(NUM_LABEL)
    label_chosen[i] = 1
    
    cost_x = 1 - tf.squeeze(tf.gather(y_conv, i, axis=1), 0)
    gradient_of_cost = tf.gradients(cost_x, x)
    x_inv = x - tf.scalar_mul(LAMBDA, tf.squeeze(gradient_of_cost, 0))
    # x_mi = np.zeros((1, x_dim))
    x_mi = np.reshape(avg_digit_img,(1,x_dim))
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

        if(i % 5000 == 0):
            print('step %d, current cost is %g' % (i, cost_x_mi))

     # Make background black instead of grey
    x_mi = np.where(x_mi<0,0, x_mi)
    x_mi = np.where(x_mi>1, 1, x_mi)

    print('iteration hit:', i+1)
    check_pred = sess.run(correct, feed_dict={x: x_mi, y: [label_chosen] })
    print("Prediction for reconstructed image:", check_pred)
    return x_mi

def plot_gan_image(name, epoch, sess):
    # Generate images from noise, using the generator network.
    fig, ax = plt.subplots(3,4)
    inverted_xs = np.zeros((NUM_LABEL, x_dim))
    for i in range(NUM_LABEL):        
        # Desired label
        d_label = np.zeros([gan_batch_size, NUM_LABEL])
        d_label[:, i] = 1
        # Noise input.
        z = np.random.uniform(-1., 1., size=[gan_batch_size, noise_dim])
        g = sess.run([gen_sample], feed_dict={gen_input: z, desired_label: d_label})
        g = np.reshape(g, (gan_batch_size, x_dim))
        avg_g = np.mean(g, axis=0)

        inverted_xs[i] = avg_g
    
        # Make background black instead of grey
        avg_g = np.where(avg_g<0, 0, avg_g)
        avg_g = np.where(avg_g>1, 1, avg_g)

        # for j in range(2):
        row = i//4
        col = i%4
        ax[row][col].imshow(np.reshape(avg_g,(28, 28)), cmap="gray", origin='lower')

    plt.savefig(name+epoch+'.png')
    plt.close()
    return inverted_xs

def train(beta, model_l2, test, load_model):
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            # Train the Classifier First
            if load_model:
                saver.restore(sess, 'tmp/cnn_mnist_model_beta%d.ckpt'%beta)
                print("Classifier Model restored.")
                test_acc = sess.run([accuracy], feed_dict={x: digits_x_test[:200,:], y: digits_y_test_one_hot[:200,:]})
            else:
                sess.run(iterator.initializer, feed_dict = {features: digits_x_train, labels: y_train_one_hot, batch_size: gan_batch_size, sample_size: 60000})
                lr = 1e-4
                batch_per_epoch = int(digits_x_train.shape[0]/gan_batch_size)
                for i in range(class_epoch):   
                    for j in range(batch_per_epoch):
                        batch = sess.run(next_batch)
                        model_optimizer.run(feed_dict={ x: batch[0], y: batch[1], learning_rate: lr})
                        # inverter_optimizer.run(feed_dict={ x: batch[0], y: batch[1]})
                    if i%20 == 0:
                        lr = lr/2.
                    train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1], learning_rate: lr})       
                    print('Epoch %d, training accuracy %g' % (i, train_accuracy))       

                test_acc = sess.run(accuracy, feed_dict={x: digits_x_test[:1000,:], y: y_test_one_hot[:1000,:]})
                print("test acc:", test_acc)
                save_path = saver.save(sess, 'tmp/cnn_mnist_model_beta%d.ckpt'%beta)
                print("Model saved in path: %s" % save_path)

            # Train Fredrickson MI
            fig, ax = plt.subplots(3,4)
            inverted_xs = np.zeros((NUM_LABEL, x_dim))
            ssims = np.zeros(NUM_LABEL)
            for i in range(NUM_LABEL):
                print("i = %d" % i)
                inverted_xs[i] = fred_mi(i, y_ml, sess, 1)[0]
                row = i//4
                col = i%4
                ax[row][col].imshow(np.reshape(inverted_xs[i], (28, 28)), cmap="gray", origin='lower')
                ssims[i]= i #compare_ssim(np.reshape(avg_imgs[i], (28, 28)), np.reshape(inverted_xs[i], (28, 28)), data_range=1.0 - 0.0)        
            plt.savefig('comparison/fred/avg_fred/fred_mi_%g_beta_%g_%s.png'%(beta,model_l2,test))
            dis = inverted_xs - avg_imgs
            l2_dis = np.linalg.norm(dis,ord=2,axis=1)
            fred_avg_dis = np.mean(l2_dis)
            fred_avg_ssim = np.mean(ssims)


            # Train GAN
            # Initialize Aux dataset for GAN train
            lr = 1e-4
            sess.run(iterator.initializer, feed_dict = {features: aux_x_data, labels: aux_y_data, batch_size: gan_batch_size, sample_size: 40000})            
            batch_per_epoch = int(aux_y_data.shape[0]/gan_batch_size)
            for i in range(gan_epoch):   
                for j in range(batch_per_epoch):           
                    # Sample random noise 
                    batch = sess.run(next_batch)
                    z = np.random.uniform(-1., 1., size=[gan_batch_size, noise_dim])
                    #! Train Discriminator
                    train_disc.run(feed_dict={aux_x: batch[0], gen_input: z, desired_label: batch[1], learning_rate: lr})
                    # if i % 5 == 0:
                    train_gen.run(feed_dict={aux_x: batch[0], gen_input: z, desired_label: batch[1], learning_rate: lr})

                if i%10 == 0:
                    lr = lr/2.
                gl,dl,cl = sess.run([gen_loss, disc_loss, gan_class_loss], feed_dict={aux_x: batch[0],    gen_input: z, desired_label: batch[1], learning_rate: lr})
                print('Epoch %i: Generator Loss: %g, Discriminator Loss: %g, Classification loss: %g' % (i, gl, dl, cl))

                inverted_xs = plot_gan_image('comparison/gan/cnn_gan_mnist/gan_out'+test+'iter', str(i), sess)

            inverted_xs = plot_gan_image('comparison/gan/gan_out',str(1000000), sess)
            ssims = np.zeros(NUM_LABEL)
            for i in range(NUM_LABEL):
                ssims[i]= i #compare_ssim(np.reshape(avg_imgs[i], (28, 28)), np.reshape(inverted_xs[i], (28, 28)), data_range=1.0 - 0.0)
            dis = inverted_xs - avg_imgs
            l2_dis = np.linalg.norm(dis,ord=2,axis=1)
            gan_avg_dis = np.mean(l2_dis)
            gan_avg_ssim = np.mean(ssims)

            return test_acc, gan_avg_dis, gan_avg_ssim, fred_avg_dis, fred_avg_ssim

if __name__ == '__main__':
    test = sys.argv[1]
    # test = 'digits'
    print('train cnn_gan_mist '+test)
    
    if test == 'letters':
        betas = [0]
        # betas = [0, 5, 10, 20, 30, 40, 60, 70, 80, 90, 100, 120]
        # betas = [0, 0.01, 0.1, 0.5, 1., 2., 5., 7., 10., 15., 20.]
        l2_coef = 0.0001
        # Use letters as aux
        load_m = False
        aux_x_data = letters_x_train
        aux_y_data = digits_y_train
        print("aux data size is ", aux_x_data.shape[0])
        aux_y_data = one_hot(digits_y_train, 10)
        aux_y_data = aux_y_data[:aux_x_data.shape[0], :]

        gan_distances = np.zeros(len(betas))
        gan_ssims = np.zeros(len(betas))
        acc = np.zeros(len(betas))
        fred_distances = np.zeros(len(betas))
        fred_ssims = np.zeros(len(betas))
        i = 0
        for beta in betas:
            acc[i], gan_distances[i], gan_ssims[i], fred_distances[i], fred_ssims[i] = train(beta, l2_coef, test+str(beta), load_m)    
            i += 1
        np.save('comparison/temp/cnn_beta_dis_gan_aiden_aux_letters', gan_distances)
        np.save('comparison/temp/cnn_beta_acc_aiden_letters', acc)
        np.save('comparison/temp/cnn_beta_ssim_gan_aiden_aux_letters', gan_ssims)
        np.save('comparison/temp/cnn_beta_dis_fred_aiden_aux_letters', fred_distances)
        np.save('comparison/temp/cnn_beta_ssim_fred_aiden_aux_letters', fred_ssims)
        # gan_distances = np.load('comparison/temp/beta_dis_gan_aiden_aux_letters.npy')
        # acc = np.load('comparison/temp/beta_acc_aiden_letters.npy')
        # gan_ssims = np.load('comparison/temp/beta_ssim_gan_aiden_aux_letters.npy')
        # fred_distances = np.load('comparison/temp/beta_dis_fred_aiden_aux_letters.npy')
        # fred_ssims = np.load('comparison/temp/beta_ssim_fred_aiden_aux_letters.npy')
        plt.close()
        plt.plot(betas, gan_distances, label='gan_distances')
        plt.plot(betas, fred_distances, label='fred_distances')
        plt.xlabel('model beta coefficient')
        plt.ylabel('sq distance between mi and avg')
        plt.legend(loc='best')
        plt.title('cnn_beta_vs_sq_dis_aiden_avgimg_aux_letters')
        plt.savefig('comparison/cnn_beta_vs_sq_dis_aiden_avgimg_aux_letters.png')
        plt.close()
        plt.plot(betas, acc)
        plt.xlabel('model beta coefficient')
        plt.ylabel('accuracy')
        plt.legend(loc='best')
        plt.title('cnn_beta_vs_accuracy_aiden_avgimg_aux_letters')
        plt.savefig('comparison/cnn_beta_vs_accuracy_aiden_avgimg_aux_letters.png')
        plt.close()
        plt.plot(betas, gan_ssims, label='gan_ssims')
        plt.plot(betas, fred_ssims, label='fred_ssims')
        plt.xlabel('model beta coefficient')
        plt.ylabel('ssim between mi and avg')
        plt.legend(loc='best')
        plt.title('cnn_beta_vs_ssims_aiden_avgimg_aux_letters')
        plt.savefig('comparison/cnn_beta_vs_ssims_aiden_avgimg_aux_letters.png')
        
    if test == 'digits':
        betas = [0]
        # betas = [0, 5, 10, 20, 30, 40, 60, 70, 80, 90, 100, 120]
        # betas = [0, 0.01, 0.1, 0.5, 1., 2., 5., 7., 10., 15., 20.]
        l2_coef = 0.0001
        # Use letters as aux
        load_m = False
        aux_x_data = digits_x_train
        aux_y_data = digits_y_train
        print("aux data size is ", aux_x_data.shape[0])
        aux_y_data = one_hot(digits_y_train, 10)
        # aux_y_data = aux_y_data[:aux_x_data.shape[0], :]

        gan_distances = np.zeros(len(betas))
        gan_ssims = np.zeros(len(betas))
        acc = np.zeros(len(betas))
        fred_distances = np.zeros(len(betas))
        fred_ssims = np.zeros(len(betas))
        i = 0
        for beta in betas:
            acc[i], gan_distances[i], gan_ssims[i], fred_distances[i], fred_ssims[i] = train(beta, l2_coef, test+str(beta), load_m)    
            i += 1
        np.save('comparison/temp/cnn_beta_dis_gan_aiden_aux_digits', gan_distances)
        np.save('comparison/temp/cnn_beta_acc_aiden_letters', acc)
        np.save('comparison/temp/cnn_beta_ssim_gan_aiden_aux_digits', gan_ssims)
        np.save('comparison/temp/cnn_beta_dis_fred_aiden_aux_digits', fred_distances)
        np.save('comparison/temp/cnn_beta_ssim_fred_aiden_aux_digits', fred_ssims)
        # gan_distances = np.load('comparison/temp/beta_dis_gan_aiden_aux_letters.npy')
        # acc = np.load('comparison/temp/beta_acc_aiden_letters.npy')
        # gan_ssims = np.load('comparison/temp/beta_ssim_gan_aiden_aux_letters.npy')
        # fred_distances = np.load('comparison/temp/beta_dis_fred_aiden_aux_letters.npy')
        # fred_ssims = np.load('comparison/temp/beta_ssim_fred_aiden_aux_letters.npy')
        plt.close()
        plt.plot(betas, gan_distances, label='gan_distances')
        plt.plot(betas, fred_distances, label='fred_distances')
        plt.xlabel('model beta coefficient')
        plt.ylabel('sq distance between mi and avg')
        plt.legend(loc='best')
        plt.title('cnn_beta_vs_sq_dis_aiden_avgimg_aux_letters')
        plt.savefig('comparison/cnn_beta_vs_sq_dis_aiden_avgimg_aux_digits.png')
        plt.close()
        plt.plot(betas, acc)
        plt.xlabel('model beta coefficient')
        plt.ylabel('accuracy')
        plt.legend(loc='best')
        plt.title('cnn_beta_vs_accuracy_aiden_avgimg_aux_letters')
        plt.savefig('comparison/cnn_beta_vs_accuracy_aiden_avgimg_aux_digits.png')
        plt.close()
        plt.plot(betas, gan_ssims, label='gan_ssims')
        plt.plot(betas, fred_ssims, label='fred_ssims')
        plt.xlabel('model beta coefficient')
        plt.ylabel('ssim between mi and avg')
        plt.legend(loc='best')
        plt.title('cnn_beta_vs_ssims_aiden_avgimg_aux_letters')
        plt.savefig('comparison/cnn_beta_vs_ssims_aiden_avgimg_aux_digits.png')
         
