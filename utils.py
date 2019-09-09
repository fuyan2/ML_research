import gzip
import struct
import array
import numpy as np

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
