"""
We get data set taken by laser and plot
"""

from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


X = pd.read_csv('/home/user/catkin_ws/src/machine_learning_course/dataset/example_laser_data_points.csv')
X = np.asarray(X)

plt.figure(figsize=(7, 7))
plt.scatter(X[:, 0], X[:, 1], 
            c='white', marker='o', edgecolor='black', s=50)
plt.grid()
plt.tight_layout()
plt.title('Data set')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.show()