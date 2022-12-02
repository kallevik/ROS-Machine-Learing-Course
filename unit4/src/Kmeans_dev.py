import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import random

class Kmeans_dev:
    """
    class definition
    """
    
    def __init__(self,X,K):
        """
        class constructor
        """
        self.X = X
        self.output = {}
        self.centroids = np.array([]).reshape(self.X.shape[1],0)
        self.K = K
        self.m = self.X.shape[0]
        
    
    def start_centroid_pos(self, X, K):
        """
        Random initialization of K centroids.
        """
        m,n = X.shape[0], X.shape[1]
        centroids = np.zeros((K,n))

        for i in range(K):
        #for i in range(1,K+1,1):
            centroids[i] = X[np.random.randint(0,m),:]

        return centroids
    

    def fit(self,n_iter):
        """
        Method to train the data set (position of centroids()
        """
        #randomly Initialize the centroids (callstart_centroid_pos() )
        self.centroids=self.start_centroid_pos(self.X,self.K)
        
        #compute Euclidian distances and assign clusters
        for n in range(n_iter):
            EuclidianDistance=np.array([]).reshape(self.m,0)
            for k in range(self.K):
                tempDist=np.sum((self.X-self.centroids[:,k])**2,axis=1)
                EuclidianDistance=np.c_[EuclidianDistance,tempDist]
            C=np.argmin(EuclidianDistance,axis=1)+1
            
            #adjust the centroids
            Y={}
            for k in range(self.K):
                Y[k+1]=np.array([]).reshape(2,0)
            for i in range(self.m):
                Y[C[i]]=np.c_[Y[C[i]],self.X[i]]
        
            for k in range(self.K):
                Y[k+1] = Y[k+1].T
            for k in range(self.K):
                self.centroids[:,k] = np.mean(Y[k+1],axis = 0)
                
            self.output=Y
            
    
    def predict(self):
        """
        Return of data set adherence to certain cluster
        """
        
        return self.output,self.centroids.T

# data set
X = np.array([[0.3,8.3 ], [3, 8], [2, 9],[0.3, 8.9],[1.7, 9.7 ],
              [0.9, 10.5], [10.3, 2.1],[10, 2],[7, 7 ], [6.9, 6.5],
              [6, 6],[1, 2], [1.5, 1.8], [5, 8 ], [8, 8], [1, 0.6],
              [9,11],[12, 5], [4.5, 4.8], [4.5, 3 ], [2, 8], [9, 3], [9,7]])
"""
Plot the data set
"""

plt.figure(figsize=(8, 8))

plt.scatter(X[:,0],X[:,1])
plt.title('Data set')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.show()

"""
Run k-means algorithm on given data set. Printing the output (position of centrod) and adherence of point to
certain cluster
"""

K= 2 # number of cluster you would like to create

#creation of class object and training (fit)
kmeans=Kmeans_dev(X,K)
kmeans.fit(50)

color=['blue','green']
labels=['cluster1','cluster2']

fig, axs = plt.subplots(5,2, figsize=(14, 28), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

#you print the test results iteration by iteration (in order to show how the centroids moves)

for i in range(10):
    kmeans=Kmeans_dev(X,K)

    kmeans.fit(1)

    Output,Centroids=kmeans.predict()
    
    for k in range(K):
        axs[i].scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
    axs[i].scatter(Centroids[:,0],Centroids[:,1],s=300,c='red',label='Centroids', marker='*')
    
    axs[i].set_title("Centroids movement. Iteration "+ str(1+i))
    
plt.show()