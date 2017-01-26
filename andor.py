import numpy as np

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def softmax(u):
    #e = np.exp(u)
    #return e / np.sum(e)
    return 1 / (1 + np.exp(-u))

def forward(x):
    global W1
    global W2
    u1 = x.dot(W1)
    z1 = sigmoid(u1)
    u2 = z1.dot(W2)
    y = softmax(u2)
    return y, z1

def back_propagation(x, z1, y, d):
    global W1
    global W2
    delta2 = y - d
    grad_W2 = z1.T.dot(delta2)

    sigmoid_dash = z1 * (1 - z1)
    delta1 = delta2.dot(W2.T) * sigmoid_dash
    grad_W1 = x.T.dot(delta1)

    W2 -= learning_rate * grad_W2
    W1 -= learning_rate * grad_W1

W1 = np.array([[0.1, 0.3], [0.2, 0.4]])
W2 = np.array([[0.1, 0.3], [0.2, 0.4]])
print W1
print W2
learning_rate = 0.1

x = np.array([[1, 0.5]])
y, z1 = forward(x)
counter = 0
e_value = 1.0e-2
er00 = 0
er01 = 0
er10 = 0
er11 = 0

while True:
    er00 = 0
    er01 = 0
    er10 = 0
    er11 = 0

    x = np.array([[0, 0]])
    d = np.array([[0, 0]])
    y, z1 = forward(x)
    back_propagation(x, z1, y, d)
    if (abs(y[0][0] - d[0][0]) < e_value) and (abs(y[0][1] - d[0][1]) < e_value):
        er00 = 1

    x = np.array([[0, 1]])
    d = np.array([[0, 1]])
    y, z1 = forward(x)
    back_propagation(x, z1, y, d)
    if (abs(y[0][0] - d[0][0]) < e_value) and (abs(y[0][1] - d[0][1]) < e_value):
        er01 = 1

    x = np.array([[1, 0]])
    d = np.array([[0, 1]])
    y, z1 = forward(x)
    back_propagation(x, z1, y, d)
    if (abs(y[0][0] - d[0][0]) < e_value) and (abs(y[0][1] - d[0][1]) < e_value):
        er10 = 1

    x = np.array([[1, 1]])
    d = np.array([[0, 0]])
    y, z1 = forward(x)
    back_propagation(x, z1, y, d)
    if (abs(y[0][0] - d[0][0]) < e_value) and (abs(y[0][1] - d[0][1]) < e_value):
        er11 = 1

    counter += 1

    #if counter > 100000 :
    #    break
    if ( er00 == 1 and er01 == 1 and er10 == 1 and er11 == 1 ):
        break



print W1
print W2
print "Learning Counter : %d" % counter
print ''

x = np.array([[0, 0]])
y, z1 = forward(x)
print "input  : " ,x
print "output : " ,y
print ''

x = np.array([[0, 1]])
y, z1 = forward(x)
print "input  : " ,x
print "output : " ,y
print ''

x = np.array([[1, 0]])
y, z1 = forward(x)
print "input  : " ,x
print "output : " ,y
print ''

x = np.array([[1, 1]])
y, z1 = forward(x)
print "input  : " ,x
print "output : " ,y
print ''

