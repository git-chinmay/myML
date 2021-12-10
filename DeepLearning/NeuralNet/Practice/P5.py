"""
ReLU Activation Function"""

import numpy as np

# X_input = [1, 1, -3, 4]
X_input = [[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]
X_input = [[10, 0.0002, 3.0, -2.5], [-2, 5.0, -1.0, -2.0], [-1.5, -2.7, 3.3, -0.8]]


class Denselayer:
    def __init__(self, no_nurons, no_inputs):
        # self.weights = np.random.rand(no_nurons, no_inputs)
        # self.biases = np.zeros(no_nurons)
        self.weights = 0.10 * np.random.rand(no_nurons, no_inputs)
        self.biases = np.zeros((no_nurons))

    def forward(self, X_input):
        self.output = np.dot(X_input, np.transpose(self.weights)) + self.biases
        return self.output


class Reluactivation:
    def forward(self, neuronInput):
        # outputList = []
        # for val in neuronInput:
        #    if val >= 0:
        #        outputList.append(val)
        #    else:
        #        outputList.append(0)
        # return outputList
        # Same thing can be done using maximum function
        self.output = np.maximum(0, neuronInput)
        return self.output


# 4 inputs thats contant here
layer1 = Denselayer(2, 4)
layer2 = Denselayer(1, 2)
NeuronRawoutputs = layer2.forward(layer1.forward(X_input))
print(NeuronRawoutputs)
actibvation = Reluactivation()
print(actibvation.forward(NeuronRawoutputs))
