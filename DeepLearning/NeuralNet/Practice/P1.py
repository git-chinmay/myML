"""
An simple exmaple of a single nuron with three inputs and one output without any activation function.
"""
X_input = [1.2, 5.1, 2.1]
weight = [3.1, 2.1, 8.7]
bias = 2
Neuron_ouput = (
    X_input[0] * weight[0] + X_input[1] * weight[1] + X_input[2] * weight[2] + bias
)
print(Neuron_ouput)
