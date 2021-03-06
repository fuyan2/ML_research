# Only train on first 20 people to reduce classifier dimensionality
# Use the rest of 20 people as auxiliary image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import tensorflow as tf
from skimage.measure import compare_ssim
from tf_models import *
from utils import *
from snwgan import snw_Generator, snw_Discriminator
# from sklearn.datasets import fetch_olivetti_faces
# from sklearn.cluster import MiniBatchKMeans
# from sklearn import decomposition
import os
import sys
from os.path import join

inf = 1e9

tf.reset_default_graph()
tf.set_random_seed(1)

# Training Params
num_steps = 100000
learning_rate = 5e-4 #0.00002
xrow = 112
xcol = 92
x_dim = 112*92 #112*92 #10304
noise_dim = 500 #20
NUM_LABEL = 20 #10
GAN_CLASS_COE = 100 #10
gan_batch_size = 40
INV_HIDDEN = 100 #5000 
beta = 0 #1, 0.5
model_l2 = 0 #0.0001 
wasserstein = True
# cnn_gan = False
#Fredrickson Params
ALPHA = 5000
BETA = 100
LAMBDA = 0.2

one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)
cur_path = os.getcwd()
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
orl_x_train, orl_y_train, orl_x_test, orl_y_test, orl_x_aux, orl_y_aux = load_ORL()

# print('training dataset size:', orl_size)
avg_imgs = average_images(NUM_LABEL, x_dim, orl_x_train, orl_y_train)
# fig, ax = plt.subplots(4,5)
# for i in range(NUM_LABEL): 
#     row = i//5
#     col = i%5
#     ax[row][col].imshow(np.reshape(avg_imgs[i,:],(xrow,xcol)), cmap="gray")
# plt.savefig('comparison/avg_imgs')
avg_imgs = np.where(avg_imgs<0,0, avg_imgs)
avg_imgs = np.where(avg_imgs>1, 1, avg_imgs)
avg_orl_img = np.mean(orl_x_train, axis=0)
plt.imshow(np.reshape(avg_orl_img,(xrow,xcol)), cmap='gray')
plt.savefig('comparison/avg_orl_img')
print('train data size is ', orl_x_train.shape[0])
print('test data size is ', orl_x_test.shape[0])
y_train_one_hot = one_hot(orl_y_train, 20)
y_test_one_hot = one_hot(orl_y_test, 20)

################### Build The Classifier ####################
x = tf.placeholder(tf.float32, shape=[None, x_dim])
y = tf.placeholder(tf.float32, shape=[None, NUM_LABEL])
model = Classifier(NUM_LABEL, x_dim)
y_ml = model(x)
# Build Inverter Regularizer
model_weights = tf.concat([tf.reshape(model.linear_w1,[1, -1]),tf.reshape(model.linear_b1,[1, -1])], 1) #, tf.reshape(model.linear_w2,[1, -1]), tf.reshape(model.linear_b2,[1, -1])], 1)
weight_shape = int(model_weights.shape[1])
inverter = Inverter_Regularizer(NUM_LABEL, x_dim, weight_shape, INV_HIDDEN)
inv_x = inverter(y, model_weights, None)
        
# Calculate MODEL Loss
inv_loss = tf.losses.mean_squared_error(labels=x, predictions=inv_x)
class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_ml))

y_pred = tf.argmax(y_ml, 1)
correct = tf.equal(tf.argmax(y_ml, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Build Optimizer !Use model_loss
inverter_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(inv_loss, var_list=[inverter.w_model, inverter.w_label, inverter.w_out, inverter.b_in, inverter.b_out])
grad_model = tf.gradients(class_loss, [model.linear_w1, model.linear_b1])#, model.linear_w2, model.linear_b2])

#################### Build GAN Networks ############################
# Network Inputs
gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
aux_x = tf.placeholder(tf.float32, shape=[None, x_dim])
aux_label = model(aux_x)
# aux_label = tf.one_hot(tf.argmax(aux_label, 1), NUM_LABEL) 

desired_label = tf.placeholder(tf.float32, shape=[None, NUM_LABEL])

# Build G Networks
# Use CNN gan
cnn_gan = False
if cnn_gan:
    G = snw_Generator(noise_dim, NUM_LABEL, gan_batch_size)
    gen_sample_unflat = G(gen_input,desired_label)
    gen_sample = tf.reshape(gen_sample_unflat,[gan_batch_size, x_dim])
    gen_vars = [G.linear_w, G.linear_b, G.deconv_w1, G.deconv_w2, G.deconv_w3]
else:
    G = Generator(noise_dim, NUM_LABEL, x_dim, gan_batch_size)
    gen_sample = G(gen_input,desired_label)
    # G Network Variables
    gen_vars = [G.linear_w1, G.linear_b1, G.linear_w2, G.linear_b2, G.linear_w3, G.linear_b3]

gen_label = model(gen_sample)

# Build 2 D Networks (one from noise input, one from generated samples)
D = Disciminator(NUM_LABEL, x_dim, wasserstein) 
disc_real = D(aux_x, aux_label)
disc_fake = D(gen_sample, gen_label)

# D Network Variables
disc_vars = [D.linear_w1, D.linear_b1, D.linear_w2, D.linear_b2]

gen_weights = tf.concat([tf.reshape(G.linear_w1,[1, -1]), tf.reshape(G.linear_b1,[1, -1]), tf.reshape(G.linear_w2,[1, -1]), tf.reshape(G.linear_b2,[1, -1]), tf.reshape(G.linear_w3,[1, -1]), tf.reshape(G.linear_b3,[1, -1])], 1)
# dis_weights = tf.concat([tf.reshape(D.linear_w1, [1,-1]), tf.reshape(D.linear_b1, [1,-1]), tf.reshape(D.linear_w2, [1,-1]), tf.reshape(D.linear_b2, [1,-1])], 1)

# Build Loss
similarity = tf.reduce_sum(tf.multiply(aux_label, desired_label), 1, keepdims=True )
gan_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=desired_label, logits=gen_label))

# Build wasserstein Loss
def wgan_grad_pen(batch_size,x,label, G_sample):    
    lam = 1   

    eps = tf.random_uniform([batch_size,1], minval=0.0,maxval=1.0)
    x_h = eps*x+(1-eps)*G_sample
    # with tf.variable_scope("", reuse=True) as scope:
    grad_d_x_h = tf.gradients(D(x_h, label), x_h)    
    grad_norm = tf.norm(grad_d_x_h[0], axis=1, ord='euclidean')
    grad_pen = tf.reduce_mean(tf.square(grad_norm-1))
  
    return lam*grad_pen

if wasserstein:
    gen_loss = -tf.reduce_mean(disc_fake) + GAN_CLASS_COE*gan_class_loss + 1. * tf.nn.l2_loss(gen_weights) #0.007, only need when no auxiliary
    # gen_loss = -tf.reduce_mean(disc_fake) + GAN_CLASS_COE*gan_class_loss + 0.01 * tf.nn.l2_loss(gen_weights)
    disc_loss = -tf.reduce_mean(similarity*(disc_real - disc_fake)) #+ wgan_grad_pen(gan_batch_size,aux_x, aux_label, gen_sample)
    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in disc_vars] #0.01!
    train_gen = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_loss, var_list=gen_vars)
    train_disc = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_loss, var_list=disc_vars)
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
    #x_mi = np.zeros((1, x_dim))
    x_mi = np.reshape(avg_orl_img,(1,x_dim))
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
    # x_mi = np.where(x_mi<0,0, x_mi)
    # x_mi = np.where(x_mi>1, 1, x_mi)

    print('iteration hit:', i+1)
    check_pred = sess.run(correct, feed_dict={x: x_mi, y: [label_chosen] })
    print("Prediction for reconstructed image:", check_pred)
    return x_mi

def plot_gan_image(name, epoch, sess):
    # Generate images from noise, using the generator network.
    fig, ax = plt.subplots(4,5)
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
        # g = np.reshape(g, [gan_batch_size, 28, 28])
        # g = g[0,:] #only pick one image

        inverted_xs[i] = avg_g
    
        # Make background black instead of grey
        avg_g = np.where(avg_g<0, 0, avg_g)
        avg_g = np.where(avg_g>1, 1, avg_g)

        # for j in range(2):
        row = i//5
        col = i%5
        ax[row][col].imshow(np.reshape(avg_g,(xrow, xcol)), cmap="gray", origin='lower')

    plt.savefig(name+epoch+'.png')
    plt.close()
    return inverted_xs

# beta = 0
# model_l2 = 0.001
# model_loss = class_loss - beta * inv_loss + model_l2*tf.nn.l2_loss(model_weights)    
# model_optimizer = tf.train.AdamOptimizer(0.001).minimize(model_loss, var_list=[model.linear_w1, model.linear_b1])#, model.linear_w2, model.linear_b2])

def train(beta, model_l2, test, load_model):
    # Train Classifier
    if test == 'l1':
        print("beta is %.4f, l1 coe is %.4f"%(beta, model_l2))
        model_loss = class_loss - beta * inv_loss + model_l2*tf.norm(model_weights, ord=1)
    else:
        print("beta is %.4f, l2 coe is %.4f"%(beta, model_l2))
        model_loss = class_loss - beta * inv_loss + model_l2*tf.nn.l2_loss(model_weights)
    
    model_optimizer = tf.train.AdamOptimizer(5e-5).minimize(class_loss, var_list=[model.linear_w1, model.linear_b1])#, model.linear_w2, model.linear_b2])

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:

            # Run the initializer
            sess.run(init)
            # sess.run(iterator.initializer, feed_dict = {features: letters_x_train, labels: letters_y_train, batch_size: gan_batch_size, sample_size: 60000})


            # Train the Classifier First
            if load_model:
                saver.restore(sess, '/tmp/face_model_beta%d.ckpt'%beta)
                print("Classifier Model restored.")
                # test_acc, y_prediction = sess.run([accuracy, y_pred], feed_dict={x: orl_x_test, y: orl_y_test})
            else:
                sess.run(iterator.initializer, feed_dict = {features: orl_x_train, labels: y_train_one_hot, batch_size: gan_batch_size, sample_size: 60000})
                for i in range(20000):
                    batch = sess.run(next_batch)
                    model_optimizer.run(feed_dict={ x: batch[0], y: batch[1]})
                    # inverter_optimizer.run(feed_dict={ x: batch[0], y: batch[1]})
                    if i % 1000 == 0:
                        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1] })
                        test_accuracy = sess.run(accuracy, feed_dict={x: orl_x_test, y: y_test_one_hot })
                        print('Epoch %d, training accuracy %g, test accuracy %g' % (i, train_accuracy, test_accuracy))        

                test_acc, y_prediction = sess.run([accuracy, y_pred], feed_dict={x: orl_x_test, y: y_test_one_hot})
                # test_acc, y_prediction = sess.run([accuracy, y_pred], feed_dict={x: letters_x_test, y: letters_y_test})
                print("test acc:", test_acc)
                save_path = saver.save(sess, '/tmp/face_model.ckpt')
                print("Model saved in path: %s" % save_path)

            # Train Fredrickson MI
            fig, ax = plt.subplots(4,5)
            inverted_xs = np.zeros((NUM_LABEL, x_dim))
            ssims = np.zeros(NUM_LABEL)
            for i in range(NUM_LABEL):
                print("i = %d" % i)
                inverted_xs[i] = fred_mi(i, y_ml, sess, 1)[0]
                row = i//5
                col = i%5
                ax[row][col].imshow(np.reshape(inverted_xs[i], (112, 92)), cmap="gray")
                ssims[i]= compare_ssim(np.reshape(avg_imgs[i], (112, 92)), np.reshape(inverted_xs[i], (112, 92)), data_range=1.0 - 0.0)        
            
            plt.savefig('comparison/fred/fred_mi_%gbeta_%g_%s.png'%(beta,model_l2,test))
            dis = inverted_xs - avg_imgs
            l2_dis = np.linalg.norm(dis,ord=2,axis=1)
            fred_avg_dis = np.mean(l2_dis)
            fred_avg_ssim = np.mean(ssims)


            # Train GAN
            # Initialize Aux dataset for GAN train
            sess.run(iterator.initializer, feed_dict = {features: aux_x_data, labels: aux_y_data, batch_size: gan_batch_size, sample_size: 40000})            
            for i in range(120000):            
                # Sample random noise 
                batch = sess.run(next_batch)
                z = np.random.uniform(-1., 1., size=[gan_batch_size, noise_dim])

                #! Train Discriminator
                train_disc.run(feed_dict={aux_x: batch[0],    gen_input: z, desired_label: batch[1]})
                if i % 5 == 0:
                    train_gen.run(feed_dict={aux_x: batch[0],    gen_input: z, desired_label: batch[1]})
                if i % 10000 == 0:
                    gl,dl,cl = sess.run([gen_loss, disc_loss, gan_class_loss], feed_dict={aux_x: batch[0],    gen_input: z, desired_label: batch[1]})
                    print('Epoch %i: Generator Loss: %f, Discriminator Loss: %f, Classification loss: %f' % (i, gl, dl, cl))

                    inverted_xs = plot_gan_image('comparison/gan/gan_out'+test+'iter', str(i), sess)

            inverted_xs = plot_gan_image('comparison/gan/log_gan_face/gan_out',str(120000), sess)
            ssims = np.zeros(NUM_LABEL)
            for i in range(NUM_LABEL):
                ssims[i]= compare_ssim(np.reshape(avg_imgs[i], (xrow, xcol)), np.reshape(inverted_xs[i], (xrow, xcol)), data_range=1.0 - 0.0)
            dis = inverted_xs - avg_imgs
            l2_dis = np.linalg.norm(dis,ord=2,axis=1)
            gan_avg_dis = np.mean(l2_dis)
            gan_avg_ssim = np.mean(ssims)

            return test_acc, gan_avg_dis, gan_avg_ssim, fred_avg_dis, fred_avg_ssim
            # if test == 'l2' or test == 'l1' or test == 'beta':
            #     return avg_dis, test_acc, avg_ssim
            # else:
            #     return avg_dis, avg_ssim

if __name__ == '__main__':
    test = sys.argv[1]

    if test == 'avg_img':
        betas = [0, 5, 10, 20, 30, 40, 60, 70, 80, 90, 100, 120]
        l2_coef = 0.0001
        load_m = False

        # use avg image as aux
        aux_x_data = np.repeat(np.reshape(avg_orl_img,[1,x_dim]),orl_y_train.shape[0], axis=0)
        aux_y_data = orl_y_train
        aux_y_data = one_hot(orl_y_train, 20)
        distances, ssims = train(betas, l2_coef, test, load_m)

        gan_distances = np.zeros(len(betas))
        gan_ssims = np.zeros(len(betas))
        acc = np.zeros(len(betas))
        fred_distances = np.zeros(len(betas))
        fred_ssims = np.zeros(len(betas))
        i = 0
        for beta in betas:
            acc[i], gan_distances[i], gan_ssims[i], fred_distances[i], fred_ssims[i] = train(beta, l2_coef, test+str(beta), load_m)    
            i += 1
        np.save('comparison/temp/beta_dis_gan_aiden_aux_avgface', gan_distances)
        np.save('comparison/temp/beta_acc_aiden_avgface', acc)
        np.save('comparison/temp/beta_ssim_gan_aiden_aux_avgface', gan_ssims)
        np.save('comparison/temp/beta_dis_fred_aiden_aux_avgface', fred_distances)
        np.save('comparison/temp/beta_ssim_fred_aiden_aux_avgface', fred_ssims)
        # gan_distances = np.load('comparison/temp/beta_dis_gan_aiden_aux_others.npy')
        # acc = np.load('comparison/temp/beta_acc_aiden_letters.npy')
        # gan_ssims = np.load('comparison/temp/beta_ssim_gan_aiden_aux_others.npy')
        # fred_distances = np.load('comparison/temp/beta_dis_fred_aiden_aux_others.npy')
        # fred_ssims = np.load('comparison/temp/beta_ssim_fred_aiden_aux_others.npy')
        plt.close()
        plt.plot(betas, gan_distances, label='gan_distances')
        plt.plot(betas, fred_distances, label='fred_distances')
        plt.xlabel('model beta coefficient')
        plt.ylabel('sq distance between mi and avg')
        plt.legend(loc='best')
        plt.savefig('comparison/beta_vs_sq_dis_aiden_avgimg_aux_avgface.png')
        plt.close()
        plt.plot(betas, acc)
        plt.xlabel('model beta coefficient')
        plt.ylabel('accuracy')
        plt.legend(loc='best')
        plt.savefig('comparison/beta_vs_accuracy_aiden_avgimg_aux_avgface.png')
        plt.close()
        plt.plot(betas, gan_ssims, label='gan_ssims')
        plt.plot(betas, fred_ssims, label='fred_ssims')
        plt.xlabel('model beta coefficient')
        plt.ylabel('ssim between mi and avg')
        plt.legend(loc='best')
        plt.savefig('comparison/beta_vs_ssims_aiden_avgimg_aux_avgface.png')

    elif test == 'no_aux':
        betas = 0
        l2_coef = 0.0001
        load_m = False
        aux_x_data = np.repeat(np.reshape(avg_orl_img,[1,x_dim]),orl_y_train.shape[0], axis=0)
        aux_y_data = orl_y_train
        print("aux data size is ", aux_x_data.shape[0])
        aux_y_data = one_hot(orl_y_train, 10)
        distances, ssims = train(betas, l2_coef, test, load_m)

    elif test == 'others':
        print('train log_gan_face others')
        betas = [0, 5, 10, 20, 30, 40, 60, 70, 80, 90, 100, 120]
        l2_coef = 0.0001
        load_m = False

        # Use other people as aux    
        aux_x_data = orl_x_aux
        aux_y_data = orl_y_train
        print("aux data size is ", aux_x_data.shape[0])
        aux_y_data = one_hot(orl_y_train, 20)
        aux_x_data = aux_x_data[:aux_y_data.shape[0], :]

        gan_distances = np.zeros(len(betas))
        gan_ssims = np.zeros(len(betas))
        acc = np.zeros(len(betas))
        fred_distances = np.zeros(len(betas))
        fred_ssims = np.zeros(len(betas))
        i = 0
        for beta in betas:
            acc[i], gan_distances[i], gan_ssims[i], fred_distances[i], fred_ssims[i] = train(beta, l2_coef, test+str(beta), load_m)    
            i += 1
        np.save('comparison/temp/beta_dis_gan_aiden_aux_others', gan_distances)
        np.save('comparison/temp/beta_acc_aiden_aux_others', acc)
        np.save('comparison/temp/beta_ssim_gan_aiden_aux_others', gan_ssims)
        np.save('comparison/temp/beta_dis_fred_aiden_aux_others', fred_distances)
        np.save('comparison/temp/beta_ssim_fred_aiden_aux_others', fred_ssims)
        # gan_distances = np.load('comparison/temp/beta_dis_gan_aiden_aux_others.npy')
        # acc = np.load('comparison/temp/beta_acc_aiden_letters.npy')
        # gan_ssims = np.load('comparison/temp/beta_ssim_gan_aiden_aux_others.npy')
        # fred_distances = np.load('comparison/temp/beta_dis_fred_aiden_aux_others.npy')
        # fred_ssims = np.load('comparison/temp/beta_ssim_fred_aiden_aux_others.npy')
        plt.close()
        plt.plot(betas, gan_distances, label='gan_distances')
        plt.plot(betas, fred_distances, label='fred_distances')
        plt.xlabel('model beta coefficient')
        plt.ylabel('sq distance between mi and avg')
        plt.legend(loc='best')
        plt.savefig('comparison/beta_vs_sq_dis_aiden_avgimg_aux_others.png')
        plt.close()
        plt.plot(betas, acc)
        plt.xlabel('model beta coefficient')
        plt.ylabel('accuracy')
        plt.legend(loc='best')
        plt.savefig('comparison/beta_vs_accuracy_aiden_avgimg_aux_others.png')
        plt.close()
        plt.plot(betas, gan_ssims, label='gan_ssims')
        plt.plot(betas, fred_ssims, label='fred_ssims')
        plt.xlabel('model beta coefficient')
        plt.ylabel('ssim between mi and avg')
        plt.legend(loc='best')
        plt.savefig('comparison/beta_vs_ssims_aiden_avgimg_aux_others.png')
