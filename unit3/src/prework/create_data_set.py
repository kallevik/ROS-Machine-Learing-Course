import numpy as np
import matplotlib.pyplot as plt

def create_data_set(points, classes):
    
    X = np.zeros((points*classes, 2)) #Array of Rows=points*classes, Columns=2 
    y = np.zeros(points*classes, dtype='uint8') #Array with points*classes elements
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.5
        X[ix] = np.c_[r*np.sin(t*1.0), r*np.cos(t*1.0)]
        y[ix] = class_number
    return X, y

plt.figure(figsize=(12,10))
X, y = create_data_set(400, 2)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.xlabel("X", fontsize=16)
plt.ylabel("Y", fontsize=16)
plt.title("Data set ", fontsize=18)
plt.show()