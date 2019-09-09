import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import tensorflow as tf
from skimage.measure import compare_ssim
from models import *
from utils import *
from snwgan import snw_Generator, snw_Discriminator
inf = 1e9

tf.reset_default_graph()
tf.set_random_seed(1)

# Training Params
num_steps = 100000
learning_rate = 0.00002 #0.00002
x_dim = 10304
noise_dim = 128 #20
NUM_LABEL = 20 #10
GAN_CLASS_COE = 100 #10
gan_batch_size = 1000
num_data = 10000
INV_HIDDEN = 100
beta = 0 #1, 0.5
model_l2 = 0 #0.0001 
wasserstein = True
# cnn_gan = False
#Fredrickson Params
ALPHA = 100000
BETA = 5
GAMMA = 1
LAMBDA = 0.06
input_desired_label = 1

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
face_x_train, face_x_test, face_y_train, face_y_test = data_segmentation('data/ORL_faces.npz', NUM_LABEL)

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
# digits_y_train = one_hot(digits_y_train, 10) #10!
# digits_y_test = one_hot(digits_y_test, 10)
# aux_y_data = one_hot(aux_y_data, 10)

# letters_y_train = one_hot(letters_y_train, 26) #10!
# letters_y_test = one_hot(letters_y_test, 26)
# aux_y_data = one_hot(aux_y_data, 26)

################### Build The Classifier ####################
x = tf.placeholder(tf.float32, shape=[None, x_dim])
y = tf.placeholder(tf.float32, shape=[None, NUM_LABEL])
model = Classifier(NUM_LABEL, x_dim)
y_ml = model(x)
# Build Inverter Regularizer
model_weights = tf.concat([tf.reshape(model.linear_w1,[1, -1]),tf.reshape(model.linear_b1,[1, -1])], 1) #, tf.reshape(model.linear_w2,[1, -1]), tf.reshape(model.linear_b2,[1, -1])], 1)
weight_shape = int(model_weights.shape[1])
inverter = Inverter_Regularizer(weight_shape, INV_HIDDEN)
inv_x = inverter(y, model_weights)
        
# Calculate MODEL Loss
inv_loss = tf.losses.mean_squared_error(labels=x, predictions=inv_x)
class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_ml))

y_pred = tf.argmax(y_ml, 1)
correct = tf.equal(tf.argmax(y_ml, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Build Optimizer !Use model_loss
inverter_optimizer = tf.train.AdamOptimizer(0.001).minimize(inv_loss, var_list=[inverter.w_model, inverter.w_label, inverter.w_out, inverter.b_in, inverter.b_out])
grad_model = tf.gradients(class_loss, [model.linear_w1, model.linear_b1])#, model.linear_w2, model.linear_b2])

#################### Build GAN Networks ############################
# Network Inputs
gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
aux_x = tf.placeholder(tf.float32, shape=[None, x_dim])
aux_label = model(aux_x)
# aux_label = tf.one_hot(tf.argmax(aux_label, 1), NUM_LABEL) 
if input_desired_label:
    desired_label = tf.placeholder(tf.float32, shape=[None, NUM_LABEL])
else:
    desired_label = aux_label

# Build G Networks
# Use CNN gan
cnn_gan = False
if cnn_gan:
    G = snw_Generator(noise_dim, NUM_LABEL, gan_batch_size)
    gen_sample_unflat = G(gen_input,desired_label)
    gen_sample = tf.reshape(gen_sample_unflat,[gan_batch_size, x_dim])
    gen_vars = [G.linear_w, G.linear_b, G.deconv_w1, G.deconv_w2, G.deconv_w3]
else:
    G = Generator(nnoise_dim, NUM_LABEL, x_dim, gan_batch_size)
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
gan_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=desired_label, logits=gen_label)) + 0.01 * tf.nn.l2_loss(gen_weights) #0.007, only need when no auxiliary

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
    gen_loss = -tf.reduce_mean(similarity*disc_fake) + GAN_CLASS_COE*gan_class_loss
    disc_loss = -tf.reduce_mean(disc_real) + tf.reduce_mean(disc_fake) #+ wgan_grad_pen(gan_batch_size,aux_x, aux_label, gen_sample)
    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in disc_vars] #0.01!
else:  
    gen_loss = -tf.reduce_mean(similarity*tf.log(tf.maximum(0.00001, disc_fake))) + GAN_CLASS_COE*gan_class_loss
    disc_loss = -tf.reduce_mean(tf.log(tf.maximum(0.0000001, disc_real)) + tf.log(tf.maximum(0.0000001, 1. - disc_fake))) 

# Create training operations !
if wasserstein:
    # train_gen = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gan_class_loss, var_list=gen_vars)
    train_gen = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_loss, var_list=gen_vars)
    train_disc = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_loss, var_list=disc_vars)
else:
    train_gen = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(gen_loss, var_list=gen_vars)
    train_disc = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(disc_loss, var_list=disc_vars)
    
# Fredrickson Model Inversion Attack, Gradient Attack
def fred_mi(i, y_conv, sess, iterate):
    label_chosen = np.zeros(NUM_LABEL)
    label_chosen[i] = 1
    
    cost_x = 1 - tf.squeeze(tf.gather(y_conv, i, axis=1), 0)
    gradient_of_cost = tf.gradients(cost_x, x)
    x_inv = x - tf.scalar_mul(LAMBDA, tf.squeeze(gradient_of_cost, 0))
    x_mi = np.zeros((1, x_dim))
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
    fig, ax = plt.subplots(NUM_LABEL)
    inverted_xs = np.zeros((NUM_LABEL, x_dim))
    for i in range(NUM_LABEL):        
        # Desired label
        d_label = np.zeros([gan_batch_size, NUM_LABEL])
        d_label[:, i] = 1
        # Noise input.
        z = np.random.uniform(-1., 1., size=[gan_batch_size, noise_dim])
        g = sess.run([gen_sample], feed_dict={gen_input: z, desired_label: d_label})
        g = np.reshape(g, [gan_batch_size, 28, 28])
        g = g[0,:] #only pick one image
        inverted_xs[i] = np.reshape(g, 28*28)
    
        # Make background black instead of grey
        g = np.where(g<0,0, g)
        g = np.where(g>1, 1, g)

        # for j in range(2):
        ax[i].imshow(g, cmap="gray", origin='lower')

    plt.savefig(name+epoch)
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
    
    model_optimizer = tf.train.AdamOptimizer(0.001).minimize(model_loss, var_list=[model.linear_w1, model.linear_b1])#, model.linear_w2, model.linear_b2])

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
                saver.restore(sess, '/tmp/model.ckpt')
                print("Classifier Model restored.")
                # test_acc, y_prediction = sess.run([accuracy, y_pred], feed_dict={x: digits_x_test, y: digits_y_test})
            else:
                sess.run(iterator.initializer, feed_dict = {features: digits_x_train, labels: y_train_one_hot, batch_size: gan_batch_size, sample_size: 60000})
                for i in range(20000):
                    batch = sess.run(next_batch)
                    model_optimizer.run(feed_dict={ x: batch[0], y: batch[1]})
                    inverter_optimizer.run(feed_dict={ x: batch[0], y: batch[1]})
                    if i % 1000 == 0:
                        gradients, train_accuracy = sess.run([grad_model, accuracy], feed_dict={x: batch[0], y: batch[1] })
    #                     print('gradients: ', gradients)
                        print('Epoch %d, training accuracy %g' % (i, train_accuracy))        

                test_acc, y_prediction = sess.run([accuracy, y_pred], feed_dict={x: digits_x_test, y: y_test_one_hot})
                # test_acc, y_prediction = sess.run([accuracy, y_pred], feed_dict={x: letters_x_test, y: letters_y_test})
                print("test acc:", test_acc)
                save_path = saver.save(sess, '/tmp/model.ckpt')
                print("Model saved in path: %s" % save_path)

            # Train Fredrickson MI
            # fig, ax = plt.subplots(6,5)
            # inverted_xs = np.zeros((NUM_LABEL, x_dim))
            # ssims = np.zeros(NUM_LABEL)
            # for i in range(NUM_LABEL):
            # fig, ax = plt.subplots(3)
            # inverted_xs = np.zeros((3, x_dim))
            # ssims = np.zeros(3)
            # for i in range(3):
            #     print("i = %d" % i)
            #     inverted_xs[i] = fred_mi(i, y_ml, sess, 1)[0]
            #     # row = i//5
            #     # col = i%5
            #     # ax[row][col]
            #     ax[i].imshow(np.reshape(inverted_xs[i], (28, 28)), cmap="gray", origin='lower')
            #     ssims[i]= compare_ssim(np.reshape(avg_imgs[i], (28, 28)), np.reshape(inverted_xs[i], (28, 28)), data_range=1.0 - 0.0)        
            # plt.savefig('comparison/fred/fred_mi_%fbeta_%fl2coe.png'%(beta,model_l2))
            # dis = inverted_xs - avg_imgs[:3, :]
            # l2_dis = np.linalg.norm(dis,ord=2,axis=1)
            # avg_dis = np.mean(l2_dis)
            # avg_ssim = np.mean(ssims)


            # Train GAN
            # Initialize Aux dataset for GAN train
            sess.run(iterator.initializer, feed_dict = {features: aux_x_data, labels: aux_y_data, batch_size: gan_batch_size, sample_size: 40000})            
            for i in range(80000):            
                # Sample random noise 
                batch = sess.run(next_batch)
                z = np.random.uniform(-1., 1., size=[gan_batch_size, noise_dim])
                if input_desired_label:
                    #! Train Discriminator
                    train_disc.run(feed_dict={aux_x: batch[0],    gen_input: z, desired_label: batch[1]})
                    if i % 5 == 0:
                        train_gen.run(feed_dict={aux_x: batch[0],    gen_input: z, desired_label: batch[1]})
                else:
                    #! Train Discriminator
                    train_disc.run(feed_dict={aux_x: batch[0],    gen_input: z})
                    if i % 5 == 0:
                        train_gen.run(feed_dict={aux_x: batch[0],    gen_input: z})

                if i % 2000 == 0:
                    gl,dl,cl = sess.run([gen_loss, disc_loss, gan_class_loss], feed_dict={aux_x: batch[0],    gen_input: z, desired_label: batch[1]})
                    print('Epoch %i: Generator Loss: %f, Discriminator Loss: %f, Classification loss: %f' % (i, gl, dl, cl))

                    inverted_xs = plot_gan_image('comparison/gan/gan_out'+test+'iter', str(i), sess)

            inverted_xs = plot_gan_image('comparison/gan/gan_out',str(50000), sess)
            ssims = np.zeros(NUM_LABEL)
            for i in range(NUM_LABEL):
                ssims[i]= compare_ssim(np.reshape(avg_imgs[i], (28, 28)), np.reshape(inverted_xs[i], (28, 28)), data_range=1.0 - 0.0)
            dis = inverted_xs - avg_imgs
            l2_dis = np.linalg.norm(dis,ord=2,axis=1)
            avg_dis = np.mean(l2_dis)
            avg_ssim = np.mean(ssims)

            if test == 'l2' or test == 'l1' or test == 'beta':
                return avg_dis, test_acc, avg_ssim
            else:
                return avg_dis, avg_ssim
            # return 0, 0
            # # Train neural net attacker
            # for i in range(80000):            
            #     batch = sess.run(next_batch)
            #     # train_nn.run(feed_dict={target_x: batch[0], target_y:batch[1]})
            #     train_nn.run(feed_dict={target_y:batch[1]})
            #     if i % 2000 == 0:
            #         # nl = sess.run(nn_loss, feed_dict={target_x: batch[0], target_y:batch[1]})
            #         nl = sess.run(nn_loss, feed_dict={target_y:batch[1]})
            #         print('Epoch %i: neural net loss:%f' % (i, nl))                 
            #         plot_nn_image('comparison/nn/nn_out',str(i), sess)

if __name__ == '__main__':
    test = 'letters'
    
    if test == 'l1': 
        # beta = 0
        l1_coef = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        distances = np.zeros(len(l1_coef))
        acc = np.zeros(len(l1_coef))
        ssims = np.zeros(len(l1_coef))
        i = 0
        for l1 in l1_coef:
            distances[i], acc[i], ssims[i] = train(beta, l1, test)    
            i += 1
        np.save('comparison/temp/l1_dis', distances)
        np.save('comparison/temp/l1_acc', acc)
        np.save('comparison/temp/l1_ssim', ssims)
        # distances = np.load('comparison/temp/l1_dis.npy')
        # acc = np.load('comparison/temp/l1_acc.npy')
        # ssims = np.load('comparison/temp/l1_ssim.npy')
        plt.show()
        plt.plot(l1_coef, distances)
        plt.xlabel('model l1 coefficient')
        plt.ylabel('sq distance between mi and avg')
        plt.savefig('comparison/l1_coef_vs_sq_dis')
        plt.close()
        plt.plot(l1_coef, acc)
        plt.xlabel('model l1 coefficient')
        plt.ylabel('accuracy')
        plt.savefig('comparison/l1_coef_vs_accuracy')
        plt.close()
        plt.plot(l1_coef, ssims)
        plt.xlabel('model l1 coefficient')
        plt.ylabel('ssim between mi and avg')
        plt.savefig('comparison/l1_coef_vs_ssim')
        
    elif test == 'l2': 
        beta = 0
        l2_coef = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        load_data = True
        if load_data:
            distances= np.load('comparison/temp/quartertrain_l2_dis.npy')
            acc = np.load('comparison/temp/quartertrain_l2_acc.npy')
            ssims = np.load('comparison/temp/quartertrain_l2_ssim.npy')
        else:
            distances = np.zeros(len(l2_coef))
            acc = np.zeros(len(l2_coef))
            ssims = np.zeros(len(l2_coef))
            load_m = False
            i = 0
            for l2 in l2_coef:
                distances[i], acc[i], ssims[i] = train(beta, l2, test, load_m)    
                i += 1
            np.save('comparison/temp/quartertrain_l2_dis', distances)
            np.save('comparison/temp/quartertrain_l2_acc', acc)
            np.save('comparison/temp/quartertrain_l2_ssim', ssims)

        plt.close()
        plt.plot(l2_coef, distances)
        plt.xlabel('model l2 coefficient')
        plt.ylabel('sq distance between mi and avg')
        plt.savefig('comparison/l2_coef_vs_sq_dis_quartertrain')
        plt.close()
        plt.plot(l2_coef, acc)
        plt.xlabel('model l2 coefficient')
        plt.ylabel('accuracy')
        plt.savefig('comparison/l2_coef_vs_accuracy_quartertrain')
        plt.close()
        plt.plot(l2_coef, ssims)
        plt.xlabel('model l2 coefficient')
        plt.ylabel('ssim between mi and avg')
        plt.savefig('comparison/l2_coef_vs_ssim_quartertrain')

    elif test == 'auxiliary':
        betas = [0,5]
        l2_coef = 0.0001
        distances0 = np.zeros(NUM_LABEL)
        acc0 = np.zeros(NUM_LABEL)
        ssims0 = np.zeros(NUM_LABEL)
        distances5 = np.zeros(NUM_LABEL)
        acc5 = np.zeros(NUM_LABEL)
        ssims5 = np.zeros(NUM_LABEL)
        load_saved = False

        if load_saved:
            distances0 = np.load('comparison/temp/aux_dis_beta0.npy')
            ssims0 = np.load('comparison/temp/aux_ssim_beta0.npy')

        else:
            for i in range(NUM_LABEL):
            # for i in range(5):
                if i == 0:
                    load_m = True
                else:
                    load_m = True
                mask = digits_y_test <=i
                aux_x_data = digits_x_test[mask,:]
                # aux_y_data = digits_y_test[mask]
                print("aux data size is ", aux_x_data.shape[0])
                print('current auxiliary data is ', i)
                # digits_y_train = one_hot(digits_y_train, 10) #10!
                aux_y_data = one_hot(digits_y_test, 10)
                aux_y_data = aux_y_data[:aux_x_data.shape[0], :]
                distances0[i], ssims0[i] = train(betas[0], l2_coef, test+str(i), load_m)

            np.save('comparison/temp/aux_dis_beta0', distances0)
            np.save('comparison/temp/aux_ssim_beta0', ssims0)

        plt.plot(range(NUM_LABEL), distances0)
        plt.xlabel('auxiliary dataset')
        plt.ylabel('sq distance between mi and avg')
        plt.savefig('comparison/aux_vs_sq_dis0')
        plt.close()
        plt.plot(range(NUM_LABEL), ssims0)
        plt.xlabel('auxiliary dataset')
        plt.ylabel('ssim between mi and avg')
        plt.savefig('comparison/aux_vs_ssim0')

    elif test == 'letters':
        betas = 0
        l2_coef = 0.0001
        load_m = True
        aux_x_data = letters_x_train
        aux_y_data = digits_y_train
        print("aux data size is ", aux_x_data.shape[0])
        aux_y_data = one_hot(digits_y_train, 10)
        aux_y_data = aux_y_data[:aux_x_data.shape[0], :]
        distances, ssims = train(betas, l2_coef, test, load_m)

    elif test == 'avg_img':
        betas = 0
        l2_coef = 0.0001
        load_m = False
        aux_x_data = np.repeat(np.reshape(avg_digit_img,[1,x_dim]),digits_y_train.shape[0], axis=0)
        aux_y_data = digits_y_train
        print("aux data size is ", aux_x_data.shape[0])
        aux_y_data = one_hot(digits_y_train, 10)
        distances, ssims = train(betas, l2_coef, test, load_m)

    elif test == 'no_aux':
        betas = 0
        l2_coef = 0.0001
        load_m = False
        aux_x_data = np.repeat(np.reshape(avg_digit_img,[1,x_dim]),digits_y_train.shape[0], axis=0)
        aux_y_data = digits_y_train
        print("aux data size is ", aux_x_data.shape[0])
        aux_y_data = one_hot(digits_y_train, 10)
        distances, ssims = train(betas, l2_coef, test, load_m)

    elif test == 'beta':
        betas = [0, 5, 10, 20, 30, 40, 60, 80, 100]
        l2_coef = 0.0001
        # Use letters as aux
        load_m = False
        aux_x_data = letters_x_train
        aux_y_data = digits_y_train
        print("aux data size is ", aux_x_data.shape[0])
        aux_y_data = one_hot(digits_y_train, 10)
        aux_y_data = aux_y_data[:aux_x_data.shape[0], :]

        distances = np.zeros(len(betas))
        acc = np.zeros(len(betas))
        ssims = np.zeros(len(betas))
        i = 0
        for beta in betas:
            distances[i], acc[i], ssims[i] = train(beta, l2_coef, test+str(beta), load_m)    
            i += 1
        np.save('comparison/temp/beta_dis', distances)
        np.save('comparison/temp/beta_acc', acc)
        np.save('comparison/temp/beta_ssim', ssims)
        # distances = np.load('comparison/temp/beta_dis.npy')
        # acc = np.load('comparison/temp/beta_acc.npy')
        # ssims = np.load('comparison/temp/beta_ssim.npy')
        plt.show()
        plt.plot(betas, distances)
        plt.xlabel('model beta coefficient')
        plt.ylabel('sq distance between mi and avg')
        plt.savefig('comparison/beta_vs_sq_dis')
        plt.close()
        plt.plot(betas, acc)
        plt.xlabel('model beta coefficient')
        plt.ylabel('accuracy')
        plt.savefig('comparison/beta_vs_accuracy')
        plt.close()
        plt.plot(betas, ssims)
        plt.xlabel('model beta coefficient')
        plt.ylabel('ssim between mi and avg')
        plt.savefig('comparison/beta_vs_ssim')