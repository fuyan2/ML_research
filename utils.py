import gzip
import struct
import array
import numpy as np
import os
from os.path import join
import re

# Load MNIST data
def mnist(type):
    def parse_labels(filename):
            with gzip.open(filename, 'rb') as fh:
                    magic, num_data = struct.unpack(">II", fh.read(8))
                    return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
            with gzip.open(filename, 'rb') as fh:
                    magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
                    return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    train_images = parse_images('data/emnist-'+type+'-train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/emnist-'+type+'-train-labels-idx1-ubyte.gz')
    test_images    = parse_images('data/emnist-'+type+'-test-images-idx3-ubyte.gz')
    test_labels    = parse_labels('data/emnist-'+type+'-test-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels

def data_segmentation(data_path, CLASSES):
  data = np.load(data_path)

  trainX = data['trainX'] / 255
  testX = data['testX'] / 255
  trainY= data['trainY']
  testY= data['testY']

  trainY = np.eye(CLASSES)[trainY]
  testY  = np.eye(CLASSES)[testY]

  return trainX, testX, trainY, testY

def load_mnist(type):
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))        
    train_images, train_labels, test_images, test_labels = mnist(type)
    train_images = partial_flatten(train_images) / 255.0
    test_images    = partial_flatten(test_images)    / 255.0
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels

def average_images(NUM_LABEL, x_dim, images, labels):
    avg_imgs = np.zeros((NUM_LABEL, x_dim))
    for i in range(NUM_LABEL):
        imgs_for_label = images[labels == i, :]
        avg_imgs[i] = np.mean(imgs_for_label, axis=0)

    return avg_imgs 

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))
    
def load_ORL():  
    print("Reading ORL faces database")
    cur_path = os.getcwd()
    path = join(cur_path, 'data', 'att_faces', 's')
    data = np.zeros((400, 112*92))
    target = np.zeros(400)
    
    for subject in range(40):
        for image in range(10):
            im = read_pgm(join(path + str(subject + 1), str(image + 1) + ".pgm"))
            data[10 * subject + image, :] = im.flatten()
            target[10 * subject + image] = subject
      
    data = data/255
    orl_x_model = data[:200, :]
    orl_y_model = target[:200]
#    orl_dataset = fetch_olivetti_faces(data_home='/Users/yanfu/Documents/ML_research/data/scikit_learn_data', shuffle=False, random_state=0, download_if_missing=True)
#    orl_x_model = orl_dataset.data[:200, :]
#    orl_y_model = orl_dataset.target[:200]
#    
    
    #Shuffle dataset and get test and train
    s = np.arange(orl_x_model.shape[0])
    np.random.shuffle(s)
    x = orl_x_model[s]
    y = orl_y_model[s]
    train_x = x[:160, :]
    train_y = y[:160]
    test_x = x[160:, :]
    test_y = y[160:]
    x_aux = data[200:, :]
    y_aux = target[200:]
    x_aux = x_aux[s]
    y_aux = y_aux[s]
    return train_x, train_y, test_x, test_y, x_aux, y_aux 
