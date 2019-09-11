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
        self.linear_w1 = tf.Variable(glorot_init([noise_dim+NUM_LABEL, 1000]),name='glw1') #500
        self.linear_b1 = tf.Variable(tf.zeros([1000]),name='glb1')
        self.linear_w2 = tf.Variable(glorot_init([1000, 200]),name='glw2') #50
        self.linear_b2 = tf.Variable(tf.zeros([200]),name='glb2')
        self.linear_w3 = tf.Variable(glorot_init([200, x_dim]),name='glw3')
        self.linear_b3 = tf.Variable(tf.zeros([x_dim]),name='glb3')
        
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
    
def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

class Inverter_Regularizer(object):
    def __init__(self, NUM_LABEL, x_dim, weight_shape, INV_HIDDEN):
        self.w_model = tf.Variable(glorot_init([weight_shape, INV_HIDDEN]))
        self.w_label = tf.Variable(glorot_init([NUM_LABEL, INV_HIDDEN]))
        self.w_out = tf.Variable(glorot_init([INV_HIDDEN, x_dim]))
        self.b_in = tf.Variable(tf.zeros([INV_HIDDEN]))
        self.b_out = tf.Variable(tf.zeros([x_dim]))
        
    def __call__(self, y, model_weights):
        # Input Layer
        ww = tf.matmul(model_weights, self.w_model)
        wy = tf.matmul(y, self.w_label)
        wt = tf.add(wy, ww)
        hidden_layer =    tf.add(wt, self.b_in)
        rect = lrelu(hidden_layer, 0.3)
        # Output Layer
        out_layer = tf.add(tf.matmul(rect, self.w_out), self.b_out)
        rect = lrelu(out_layer, 0.3)
        return rect