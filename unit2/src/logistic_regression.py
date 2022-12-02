# Definition of class

class LogisticRegression:
    def __init__(self, lr=0.0001, num_iter=100000):
        self.lr = lr
        self.num_iter = num_iter
        self.logistic_loss = []
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
                
            if(i % 1000 == 0):
                self.logistic_loss.append(loss)
            if(i % 5000 == 0):
                print ("i:", i, "loss: ", loss)
                
                
    
    def predict_prob(self, X):
        #if self.fit_intercept:
         #   X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()
    
    
    
    
#creating a object to Logistic Regression    
model = LogisticRegression(lr=0.1, num_iter=100000)
# Now we will learn our model
model.fit(X, y)

# print the eroor while training
plt.figure(figsize=(10, 8))
i = np.arange(0, len(model.logistic_loss),1)
plt.plot(i, model.logistic_loss)
plt.xlabel("Iteration", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.show()


# print logistic regression applied to data set
plt.figure(figsize=(10, 8))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1],  label='1')
plt.legend()
x1_min, x1_max = X[:,0].min(), X[:,0].max(),
x2_min, x2_max = X[:,1].min(), X[:,1].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model.predict_prob(grid).reshape(xx1.shape)
plt.xlabel("Deviation - x direction", fontsize=16)
plt.ylabel("Deviation - y direction", fontsize=16)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='red')
plt.show()