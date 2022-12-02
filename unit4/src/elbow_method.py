"""
Definition of data set taken to run following example. 
Application of Elbow method to estimate the numer of recommended centroids.
"""

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
from sklearn.cluster import KMeans # you use sklearn to compute the "elbow" - optimal number of clusters 


# data set
X = np.array([[0.3,8.3 ], [3, 8], [2, 9],[0.3, 8.9],[1.7, 9.7 ],
              [0.9, 10.5], [10.3, 2.1],[10, 2],[7, 7 ], [6.9, 6.5],
              [6, 6],[1, 2], [1.5, 1.8], [5, 8 ], [8, 8], [1, 0.6],
              [9,11],[12, 5], [4.5, 4.8], [4.5, 3 ], [2, 8], [9, 3], [9,7]])

# now you run the estimation of clusters (Elbow method) you shoud apply for our algorithm
#you plot the the analysis

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
plt.ylabel('Within-cluster Sum of Square (WSS)')
plt.title('Elbow method to estimate number of clusters')
plt.tight_layout()

plt.show()