"""
Making NN more generic with Class objects
Making batches we can run in parallel many batches using GPUs.

Input will be list of lists
We will create Multilayer Neural Network
"""
import numpy as np

# First Layer Neuron
X_input = [[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]
weights1 = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases1 = [2.0, 3.0, 0.5]

Layer1_output = np.dot(X_input, np.transpose(weights1)) + biases1
# print(Layer1_output)
# Second Layer Neuron
# Here input willbe First Layer Neuron output
weights2 = [[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]
Layer2_output = np.dot(Layer1_output, np.transpose(weights2)) + biases2
# print(Layer2_output)

"""
Achieveing above in class object
"""


class DenseLayer:
    def __init__(self, no_of_inputs, no_of_neurons):
        # we take no of inputs as row bcz we dont have to Transpose duirng calculation in forwrd method
        self.weights = 0.10 * np.random.randn(no_of_inputs, no_of_neurons)
        # self.weights = 0.10 * np.random.randn(no_of_neurons, no_of_inputs)
        self.biases = np.zeros((1, no_of_neurons))

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.biases
        # self.output = np.dot(input, np.transpose(self.weights)) + self.biases


Layer1 = DenseLayer(4, 2)
Layer2 = DenseLayer(2, 1)
Layer1.forward(X_input)
# print(Layer1.output)
Layer2.forward(Layer1.output)
print(Layer2.output)
