# coding: utf-8

import os
import struct
from array import array

class MNIST(object):
    def __init__(self):
        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []

    def load_training(self):
        ims, labels = self.load(('./mnist/train-images-idx3-ubyte'),
                                 ('./mnist/train-labels-idx1-ubyte'))

        self.train_images = ims
        self.train_labels = labels

        return ims, labels

    def load_test(self):
        tims, tlabels = self.load(('./mnist/t10k-images-idx3-ubyte'),('./mnist/t10k-labels-idx1-ubyte'))
        self.test_images = tims
        self.test_labels = tlabels

        return tims, tlabels


    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
                    magic, size = struct.unpack(">II", file.read(8))
                    if magic != 2049:
                        raise ValueError('Magic number mismatch, expected 2049,'
                                         'got {}'.format(magic))

                    labels = array("B", file.read())

        with open(path_img, 'rb') as file:
                    magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
                    if magic != 2051:
                        raise ValueError('Magic number mismatch, expected 2051,'
                                         'got {}'.format(magic))

                    image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels

