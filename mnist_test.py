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

#normed = train_img[0] / np.linalg.norm(train_img[0])
#npdata = np.array(train_img[0])
#print normed
#print normed.sum()
#print npdata
#print npdata / 255.0

mlp = make_3Layered_perceptron.make_perceptron(n_input_units=784, n_hidden_units=50, n_output_units=10)

inputs = np.empty((0, 784), float)
targets = np.empty((0, 10), int)
#dataset = zip(test_img, test_label)

start = time.time()

# set data
#print train_img[1]
#print train_label[1]
#for i in xrange(trdata_num):
    #npdata = np.array(train_img[i])
    #inputs = np.vstack((inputs, (npdata / 255.0)))
#    labels = np.zeros(10)
#    labels[train_label[i]] = 1
#    targets = np.vstack((targets, labels))
    #sys.stdout.write('i')

#inputs = [train_img[i] for i in xrange(trdata_num)]
#targets = [train_label[i] for i in xrange(trdata_num)]
#print targets
#set_time = time.time()
inputs = train_img[0:trdata_num]
#zeros = [np.zeros(trdata_num) for i in xrange(trdata_num)]
targets = [np.insert(np.zeros(9), train_label[i], 1) for i in xrange(trdata_num)]
set_time = time.time()

# training
print '---training---'
print 'Num of training data : %d' %trdata_num
mlp.fit(inputs, targets, learning_rate=0.05, epochs=10)
learning_time = time.time()

# predict
print '---predict---'
print ''
#cnt = 0
#for i in xrange(tedata_num):
#    npdata = np.array(test_img[i])
#    npdata = list(npdata)
#    predict = mlp.predict(npdata)
#    index = np.nanargmax(predict)
#    if(index == test_label[i]):
#        cnt += 1

correct_ary = [np.nanargmax(mlp.predict(list(test_img[i]))) == test_label[i] for i in xrange(tedata_num)]

predict_time = time.time()

print '---result---'
print 'correct : %d' %sum(correct_ary)
print 'data set time(s) : %f' %(set_time - start)
print 'learning time(s) : %f' %(learning_time - set_time)
print 'predict time(s) : %f' %(predict_time - learning_time)
print '\007'
