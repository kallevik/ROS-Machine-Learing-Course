"""
We get data set taken by laser and plot
"""

from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


X = pd.read_csv('/home/user/catkin_ws/src/machine_learning_course/dataset/example_laser_data_points.csv')
X = np.asarray(X)

"""
Nnow you run the estimation of clusters (Elbow method) you shoud apply for our algorithm
You plot the the analysis 
"""

distortions = []

for i in range(1, 11):
    km = KMeans(n_clusters=i, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()

plt.show()