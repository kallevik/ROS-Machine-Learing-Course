import numpy as np
import matplotlib.pyplot as plt


def logistic(x):
    return(1/(1+np.exp(-x)))


t = np.arange(-5,5,0.1)

sig = [logistic(i) for i in t]

plt.figure(figsize=(16, 8))
plt.xlabel("Input", fontsize=16)
plt.ylabel("Sigmoid function", fontsize=16)
plt.plot(t, sig)
plt.grid()
plt.show()