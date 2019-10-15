import tensorflow as tf

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.), dtype=tf.float32)

# Linear Regression 
class NN_attacker(object):
    def __init__(self, NUM_LABEL, x_dim):
        self.linear_w1 = tf.Variable(glorot_init([NUM_LABEL, 1000]),name='glw1') #500 for mnist
        self.linear_b1 = tf.Variable(tf.zeros([1000]),name='glb1')
        self.linear_w2 = tf.Variable(glorot_init([1000, 200]),name='glw2') #50 for mnist
        self.linear_b2 = tf.Variable(tf.zeros([200]),name='glb2')
        self.linear_w3 = tf.Variable(glorot_init([200, x_dim]),name='glw3')
        self.linear_b3 = tf.Variable(tf.zeros([x_dim]),name='glb3')

    def __call__(self, y):
        linear_z1 = tf.nn.leaky_relu(tf.matmul(y,self.linear_w1) + self.linear_b1)
        linear_z2 = tf.nn.leaky_relu(tf.matmul(linear_z1,self.linear_w2) + self.linear_b2)
        out_layer = tf.matmul(linear_z2,self.linear_w3)+self.linear_b3
        return out_layer

class NN_aux_attacker(object):
    def __init__(self, NUM_LABEL, x_dim):
        self.linear_w1 = tf.Variable(glorot_init([NUM_LABEL+x_dim, 500]),name='glw1')
        self.linear_b1 = tf.Variable(tf.zeros([500]),name='glb1')
        self.linear_w2 = tf.Variable(glorot_init([500, 50]),name='glw2')
        self.linear_b2 = tf.Variable(tf.zeros([50]),name='glb2')
        self.linear_w3 = tf.Variable(glorot_init([50, x_dim]),name='glw3')
        self.linear_b3 = tf.Variable(tf.zeros([x_dim]),name='glb3')

    def __call__(self, y, x):
        x_y = tf.concat((x,y),1)
        linear_z1 = tf.nn.leaky_relu(tf.matmul(x_y,self.linear_w1) + self.linear_b1)
        linear_z2 = tf.nn.leaky_relu(tf.matmul(linear_z1,self.linear_w2) + self.linear_b2)
        out_layer = tf.matmul(linear_z2,self.linear_w3)+self.linear_b3
        return out_layer

class Generator(object):
    # G Parameters
    def __init__(self, noise_dim, NUM_LABEL, x_dim, batch_size):
        self.batch_size = batch_size
        self.linear_w1 = tf.Variable(glorot_init([noise_dim+NUM_LABEL, 500]),name='glw1') #500, 1000
        self.linear_b1 = tf.Variable(tf.zeros([500]),name='glb1')
        self.linear_w2 = tf.Variable(glorot_init([500, 50]),name='glw2') #50, 200
        self.linear_b2 = tf.Variable(tf.zeros([50]),name='glb2')
        self.linear_w3 = tf.Variable(glorot_init([50, x_dim]),name='glw3')
        self.linear_b3 = tf.Variable(tf.zeros([x_dim]),name='glb3')

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
    def __init__(self, NUM_LABEL, x_dim, wasserstein):
        self.linear_w1 = tf.Variable(glorot_init([x_dim + NUM_LABEL, 100]))
        self.linear_b1 = tf.Variable(tf.zeros([100]))
        self.linear_w2 = tf.Variable(glorot_init([100, 1]))
        self.linear_b2 = tf.Variable(tf.zeros([1]))
        self.wasserstein = wasserstein

    # Build D Graph
    def __call__(self, x, y):
        x_y = tf.concat((x,y),1)
        linear1 = tf.nn.relu(tf.matmul(x_y, self.linear_w1) + self.linear_b1)
        out = tf.matmul(linear1, self.linear_w2) + self.linear_b2
        if self.wasserstein:
            return out
        else:
            return tf.sigmoid(out)

class Classifier(object):
    def __init__(self, NUM_LABEL, x_dim):
        self.linear_w1 = tf.Variable(glorot_init([x_dim, NUM_LABEL]))
        self.linear_b1 = tf.Variable(tf.zeros([NUM_LABEL]))
        # self.linear_w2 = tf.Variable(glorot_init([100, NUM_LABEL]))
        # self.linear_b2 = tf.Variable(tf.zeros([NUM_LABEL]))

        self.training = True

    # Build D Graph
    def __call__(self, x):
        # linear1 = tf.nn.relu(tf.matmul(x, self.linear_w1) + self.linear_b1)
        # out = tf.nn.softmax(tf.matmul(linear1, self.linear_w2) + self.linear_b2)
        out = tf.nn.softmax(tf.matmul(x, self.linear_w1) + self.linear_b1)
        return out

class CNN_Classifier(object):
    def __init__(self, NUM_LABEL, x_dim):
        NUM_CHANNEL = 1 # Black white image
        FILTER_SIZE = 5 # Use 5x5 filter for all conv layer
        DEPTH_1 = 32 # num output feature maps for first layer
        DEPTH_2 = 64
        HIDDEN_UNIT = 1024
        CONV_OUT = 7
        self.NUM_LABEL = NUM_LABEL

        self.conv_w1 = tf.Variable(glorot_init([FILTER_SIZE, FILTER_SIZE, NUM_CHANNEL, DEPTH_1]))
        self.conv_w2 = tf.Variable(glorot_init([FILTER_SIZE, FILTER_SIZE, DEPTH_1, DEPTH_2]))
        self.conv_b1 = tf.Variable(tf.constant(0.1, shape=[DEPTH_1]), name='b1') # Why initialize to 0.1?
        self.conv_b2 = tf.Variable(tf.constant(0.1, shape=[DEPTH_2]), name='b2')

        self.full_w = tf.Variable(glorot_init([CONV_OUT*CONV_OUT*DEPTH_2, HIDDEN_UNIT]))
        self.full_b = tf.Variable(tf.constant(0.1, shape=[HIDDEN_UNIT]), name='full_b')

        self.out_w = tf.Variable(glorot_init([HIDDEN_UNIT,self.NUM_LABEL]))
        self.out_b = tf.Variable(tf.constant(0.1, shape=[self.NUM_LABEL]), name='out_b')

    def __call__(self,x):
        # First Conv Layer with relu activation and max pool
        conv_xw1 = tf.nn.conv2d(x,self.conv_w1,strides=[1, 1, 1, 1], padding='SAME')
        conv_z1 = tf.nn.relu(conv_xw1 + self.conv_b1)
        conv_out1 = tf.nn.max_pool(conv_z1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

        # Second Conv Layer with relu activation and max pool
        conv_xw2 = tf.nn.conv2d(conv_out1, self.conv_w2,strides=[1, 1, 1, 1], padding='SAME')
        conv_z2 = tf.nn.relu(conv_xw2 + self.conv_b2)
        conv_out2 = tf.nn.max_pool(conv_z2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
        conv_out2_flat = tf.reshape(conv_out2, [-1, 7*7*64])
        # Fully Connected Layer with Relu Activation
        full_out = tf.nn.relu(tf.matmul(conv_out2_flat, self.full_w) + self.full_b)

        # Output Layer
        y_ml = tf.nn.softmax(tf.matmul(full_out, self.out_w) + self.out_b)
        return y_ml

def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


class Inverter_Regularizer(object):
    def __init__(self, NUM_LABEL, x_dim, weight_shape, INV_HIDDEN):
        self.w_model = tf.Variable(glorot_init([weight_shape, INV_HIDDEN]))
        self.w_label = tf.Variable(glorot_init([NUM_LABEL, INV_HIDDEN]))
        self.w_aux = tf.Variable(glorot_init([x_dim, INV_HIDDEN]))
        self.w_out = tf.Variable(glorot_init([INV_HIDDEN, x_dim]))
        self.b_in = tf.Variable(tf.zeros([INV_HIDDEN]))
        self.b_out = tf.Variable(tf.zeros([x_dim]))
        
    def __call__(self, y, model_weights, aux):
        # Input Layer
        if aux == None:
            ww = tf.matmul(model_weights, self.w_model)
            wy = tf.matmul(y, self.w_label)
            wt = tf.add(wy, ww)
        else:
            ww = tf.matmul(model_weights, self.w_model)
            wy = tf.matmul(y, self.w_label)
            wo = tf.matmul(aux, self.w_aux)
            wh = tf.add(wy, ww)
            wt = tf.add(wh,wo)

        hidden_layer = tf.add(wt, self.b_in)
        rect = lrelu(hidden_layer, 0.3)
        # Output Layer
        out_layer = tf.add(tf.matmul(rect, self.w_out), self.b_out)
        rect = lrelu(out_layer, 0.3)
        return rect

class cnn_Generator:
  # Generator Parameters
  def __init__(self, noise_dim, NUM_LABEL, batch_size):
    self.batch_size = batch_size
    self.linear_w = tf.Variable(glorot_init([noise_dim+NUM_LABEL, 7*7*256]),name='glw')
    self.linear_b = tf.Variable(glorot_init([7*7*256]),name='glb')

    self.deconv_w1 = tf.Variable(glorot_init([4, 4, 128, 256]),name='gdw1')
    self.deconv_w2 = tf.Variable(glorot_init([4, 4, 64, 128]),name='gdw2')
    self.deconv_w3 = tf.Variable(glorot_init([3, 3, 1, 64]),name='gdw3')

    # self.deconv_b1 = tf.Variable(tf.constant(0.1, shape=[128]), name='gb1')
    # self.deconv_b2 = tf.Variable(tf.constant(0.1, shape=[64]), name='gb2')
    # self.deconv_b3 = tf.Variable(tf.constant(0.1, shape=[1]), name='gb3')

    self.training = True

 # Build Generator Graph
  def __call__(self, z,y):
    z_y = tf.concat([z,y],1)
    linear_h = tf.matmul(z_y,self.linear_w)+self.linear_b
    linear_h_reshape = tf.reshape(linear_h , [-1,7, 7,256])  
    deconv_xw1 = tf.nn.conv2d_transpose(linear_h_reshape, self.deconv_w1,output_shape=[self.batch_size,14,14,128], strides=[1, 2, 2, 1])
    xw1_norm = tf.layers.batch_normalization(deconv_xw1, training=self.training)
    deconv_h1 = tf.nn.leaky_relu(xw1_norm )
    deconv_xw2 = tf.nn.conv2d_transpose(deconv_h1, self.deconv_w2,output_shape=[self.batch_size,28,28,64], strides=[1, 2, 2, 1])
    xw2_norm = tf.layers.batch_normalization(deconv_xw2, training=self.training)
    deconv_h2 = tf.nn.leaky_relu(xw2_norm)
    deconv_xw3 = tf.nn.conv2d_transpose(deconv_h2, self.deconv_w3,output_shape=[self.batch_size,28,28,1], strides=[1, 1, 1, 1])
    # out_layer = tf.nn.sigmoid(deconv_xw3)
    out_layer = tf.nn.tanh(deconv_xw3)
    return out_layer

class cnn_Discriminator:
  # Discriminator Parameters
  def __init__(self, NUM_LABEL):
    self.conv_w1 = tf.get_variable('dw1', shape=[3, 3, 1, 64])
    self.conv_w2 = tf.get_variable('dw2', shape=[4, 4, 64, 128])
    self.conv_w3 = tf.get_variable('dw3', shape=[3, 3, 128, 128])
    self.conv_w4 = tf.get_variable('dw4', shape=[4, 4, 128, 256])
    # self.conv_w5 = tf.get_variable('dw5', shape=[3, 3, 256, 256])

    # self.conv_b1 = tf.Variable(tf.constant(0.1, shape=[64]), name='db1') 
    # self.conv_b2 = tf.Variable(tf.constant(0.1, shape=[128]), name='db2') 
    # self.conv_b3 = tf.Variable(tf.constant(0.1, shape=[128]), name='db3') 
    # self.conv_b4 = tf.Variable(tf.constant(0.1, shape=[256]), name='db4') 
    # self.conv_b5 = tf.Variable(tf.constant(0.1, shape=[256]), name='db5') 

    self.linear_w1 = tf.Variable(glorot_init([7*7*256+NUM_LABEL, 300]))
    self.linear_b1 = tf.Variable(glorot_init([300]))
    self.linear_w2 = tf.Variable(glorot_init([300, 1]))

    self.training = True

  # Build Discriminator Graph
  def __call__(self, x, label):
    conv_xw1 = tf.nn.conv2d(x, self.conv_w1,strides=[1, 1, 1, 1], padding='SAME')
    xw1_norm = tf.layers.batch_normalization(conv_xw1, training=self.training)
    conv_h1 = tf.nn.leaky_relu(xw1_norm)
    conv_xw2 = tf.nn.conv2d(conv_h1, self.conv_w2, strides=[1, 2, 2, 1], padding='SAME')
    xw2_norm = tf.layers.batch_normalization(conv_xw2, training=self.training)
    conv_h2 = tf.nn.leaky_relu(xw2_norm)
    conv_xw3 = tf.nn.conv2d(conv_h2, self.conv_w3, strides=[1, 1, 1, 1], padding='SAME')
    xw3_norm = tf.layers.batch_normalization(conv_xw3, training=self.training)
    conv_h3 = tf.nn.leaky_relu(xw3_norm)
    conv_xw4 = tf.nn.conv2d(conv_h3, self.conv_w4, strides=[1, 2, 2, 1], padding='SAME')
    xw4_norm = tf.layers.batch_normalization(conv_xw4, training=self.training)
    conv_h4 = tf.nn.leaky_relu(xw4_norm)
    # conv_xw5 = tf.nn.conv2d(conv_h4,spectral_normed_weight(self.conv_w5, num_iters=1, update_collection=SPECTRAL_NORM_UPDATE_OPS),strides=[1, 1, 1, 1], padding='SAME')
    # xw5_norm = tf.layers.batch_normalization(conv_xw5, training=self.training)
    # conv_h5 = tf.nn.leaky_relu(xw5_norm)
    conv_h5_flat = tf.reshape(conv_h4, [-1, 7*7*256])
    x_y = tf.concat((conv_h5_flat,label),1)
    linear1 = tf.matmul(x_y, self.linear_w1) + self.linear_b1
    linear1 = tf.nn.leaky_relu(linear1)

    out = tf.matmul(linear1, self.linear_w2)
    #! out = tf.sigmoid(out)
    return out

