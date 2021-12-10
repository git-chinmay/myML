"""
MOdelling a 4 input with 3 Neurons.Out put will be 3 element list as each Neuron will have one putput
Note:- No of Nurons = No of Biases
Each wright = No of inputs
"""
X_input = [1.0, 2.0, 3.0, 2.5]
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
bias1 = 2.0
bias2 = 3.0
bias3 = 0.5
Neuron_ouput = [
    X_input[0] * weights1[0]
    + X_input[1] * weights1[1]
    + X_input[2] * weights1[2]
    + X_input[3] * weights1[3]
    + bias1,
    X_input[0] * weights2[0]
    + X_input[1] * weights2[1]
    + X_input[2] * weights2[2]
    + X_input[3] * weights2[3]
    + bias2,
    X_input[0] * weights3[0]
    + X_input[1] * weights3[1]
    + X_input[2] * weights3[2]
    + X_input[3] * weights3[3]
    + bias3,
]
print(Neuron_ouput)
