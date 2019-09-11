#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 21:05:39 2019

@author: yanfu
"""

import re
import numpy as np
import os
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.image
 
cur_path = os.getcwd()

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
    path = join(cur_path, 'data', 'att_faces', 's')
    data = np.zeros((400, 92 * 112))
    target = np.zeros(400)
    
    for subject in range(40):
        for image in range(10):
            im = read_pgm(join(path + str(subject + 1), str(image + 1) + ".pgm"))
            data[10 * subject + image, :] = im.flatten()
            target[10 * subject + image] = subject
            
    orl_x_model = data[:200, :]
    orl_y_model = target[:200]
    
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
    return train_x, train_y, test_x, test_y, x_aux, y_aux 




if __name__ == "__main__":
    
#    image = read_pgm("/Users/yanfu/Documents/ML_research/data/att_faces/s1/1.pgm", byteorder='<')
#    pyplot.imshow(image, pyplot.cm.gray)
#    pyplot.show()
    train_x, train_y, test_x, test_y, x_aux, y_aux  = load_ORL()
    plt.imshow(np.reshape(train_x[0,:],(112,92)), cmap="gray")