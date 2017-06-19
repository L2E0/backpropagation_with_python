# -*- coding: utf-8 -*-
from WMNIST import MNIST
import make_3Layered_perceptron
import sys
import numpy as np
import time


mndata = MNIST()
train_img, train_label = mndata.load_training()
test_img, test_label = mndata.load_test()

train_img = np.array(train_img)
train_img = train_img / 255.0
test_img = np.array(test_img)
test_img = test_img / 255.0

trdata_num = 60000
tedata_num = 10000

mlp = make_3Layered_perceptron.make_perceptron(n_input_units=784, n_hidden_units=300, n_output_units=10)

inputs = np.empty((0, 784), float)
targets = np.empty((0, 10), int)
#dataset = zip(test_img, test_label)

start = time.time()

inputs = train_img[0:trdata_num]
targets = [np.insert(np.zeros(9), train_label[i], 1) for i in xrange(trdata_num)]
set_time = time.time()

# training
epochs = 500 
print '---training---'
print 'Num of training data : %d' %trdata_num
print 'Num of test data : %d' %tedata_num
print 'learning_rate : 0.05, epochs : %d\n' %epochs
for j in range(epochs):
    mlp.fit(inputs, targets, learning_rate=0.05, epochs=1)
    learning_time = time.time()
    predict_ary = [np.nanargmax(mlp.predict(list(test_img[i]))) == test_label[i] for i in xrange(tedata_num)]
    print 'epoch : %d, right predicts : %d, loss : %f' %(j+1, sum(predict_ary), (tedata_num - sum(predict_ary)) / float(tedata_num))
    
# predict
print '---predict---'
print ''


predict_time = time.time()

print '---result---'
print 'right predicts : %d' %sum(predict_ary)
print 'learning time(s) : %f' %(learning_time - set_time)
print 'loss : %f' %((tedata_num - sum(predict_ary)) / float(tedata_num))
print '\007'
