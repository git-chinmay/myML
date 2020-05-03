"""
Compact version of P2
"""

X_input = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]
finalOutput = []

for neuronWeights, neuronBiase in zip(weights, biases):
    # extract weights & biases of each neuron
    nuronOut = 0
    for weight, input in zip(neuronWeights, X_input):
        # Extraxt each weight value and input value to multiply
        nuronOut += weight * input
    # After multiplication of weights and inputs add the bias of the neuron
    nuronOut += neuronBiase
    finalOutput.append(nuronOut)

print(finalOutput)

# Performing above using numpy
import numpy as np

finalOutput = np.dot(X_input, np.transpose(weights)) + biases
# finalOutput = np.dot(weights, X_input) + biases
print(finalOutput)
