import numpy as np

def activation_function_relu(x):
    if x > 0:
        return x
    else:
        return 0 
    
    
inputs = [2.8, 4.9, 7.1, 10.1]
weights = [3.1, 1.1, 0.7, 3.3]
bias = 1.0


output_sum = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3]+ bias
neuron_output = activation_function_relu(output_sum)
print(neuron_output)

def activation_function_relu2(x):
    return np.maximum(0, x)

inputs = [2.8, 4.9, 7.1, 10.1]
weights = [3.1, 1.1, .7, 3.3]
bias = 1.0

output_sum = np.dot(weights, inputs) + bias
neuron_output = activation_function_relu2(output_sum)
print("neuron output :: ", neuron_output)