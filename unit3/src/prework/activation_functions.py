import math
import matplotlib.pyplot as plt
import numpy as np

"""
The definition of activation functions mathematically
"""
# Sigmoid Function
def sigmoid(x):
    a = []
    for i in x:
        a.append(1/(1+math.exp(-i)))
    return a


# Hyperbolic Tanjant Function
def tanh(x):
    return np.tanh(x)


# ReLU Function
def relu(x):
    b = []
    for i in x:
        if i<0:
            b.append(0)
        else:
            b.append(i)
    return b


# Leaky ReLU Function
def lrelu(x):
    b = []
    for i in x:
        if i<0:
            b.append(i/10)
        else:
            b.append(i)
    return b
  
# Determining the intervals to be created for the graph
x = np.arange(-2., 2., 0.1)
sig = sigmoid(x)
tanh = tanh(x)
relu = relu(x)
leaky_relu = lrelu(x)

# Displaying the functions
plt.figure(figsize=(12,10))
line_1, = plt.plot(x,sig, label='Sigmoid')
line_2, = plt.plot(x,tanh, label='Tanh')
line_3, = plt.plot(x,relu, label='ReLU')
line_4, = plt.plot(x,leaky_relu, label='Leaky ReLU', linestyle = '--')
plt.legend(handles=[line_1, line_2, line_3, line_4], fontsize=14)
plt.axhline(y=0, color='k', linestyle = '--')
plt.axvline(x=0, color='k', linestyle = '--')
plt.show()