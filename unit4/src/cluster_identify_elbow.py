from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# you get data set taken by laser and plot. CYLINDERS ARE DEPLOYED BY YOU!
X = pd.read_csv('/home/user/catkin_ws/src/machine_learning_course/dataset/example_laser_data_points.csv')
#/home/user/catkin_ws/src/results/laser_data_points.csv
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


"""
Now you run the estimation of clusters (Elbow method) you should apply for our algorithm
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

"""
Now you learn the model based on your own data set
"""

from sklearn.cluster import KMeans

# definition of object and setup
km = KMeans(n_clusters=5, 
            init='random', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
# we learn our model (object) with provided data set
y_km = km.fit_predict(X)

"""
Now you plot our results - cluseters and centroids
"""
plt.figure(figsize=(7, 7))
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c='lightgreen',marker='s', edgecolor='black', label='Cluster 1')
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='orange',marker='o', edgecolor='black',label='Cluster 2')
plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c='lightblue', marker='v', edgecolor='black', label='Cluster 3')
plt.scatter(X[y_km == 3, 0], X[y_km == 3, 1], s=50, c='orange',marker='>', edgecolor='black',label='Cluster 4')
plt.scatter(X[y_km == 4, 0], X[y_km == 4, 1], s=50, c='lightblue', marker='<', edgecolor='black', label='Cluster 5')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, marker='*', c='red', edgecolor='black',label='Centroid')


for i in range(len(km.cluster_centers_)):
    print ("Position of centroid (obstacle). X ::", km.cluster_centers_ [i,0], "Y ::", km.cluster_centers_[i,1])

plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()