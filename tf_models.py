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
