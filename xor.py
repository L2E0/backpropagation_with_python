import numpy as np
import make_3Layered_perceptron

mlp = make_3Layered_perceptron.make_perceptron(n_input_units=2, n_hidden_units=3, n_output_units=1)

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])

# training
mlp.fit(inputs, targets, learning_rate=0.1, epochs=100000)

# predict
print '--- predict ---'
for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
    print i, mlp.predict(i)
