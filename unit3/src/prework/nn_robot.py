import numpy as np
import matplotlib.pyplot as plt

def create_data_set(points, classes):
    """
    Creates a data set of given number of classes and
    scatters them 
    """
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.5
        X[ix] = np.c_[r*np.sin(t*1.0), r*np.cos(t*1.0)]
        y[ix] = class_number
    return X, y

X, y = create_data_set(400, 2)

N = 200 # number of points per class
D = 2 # dimensionality
K = 2 # number of classes
h = 1000 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))
step_size = 1e-0


num_examples = X.shape[0]
ii = []
error = []

#here in for-loop we deply the back propagation alghorithm (gradient descent)

for i in range(3000):
  
  # evaluate class scores, [N x K]
    hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
    scores = np.dot(hidden_layer, W2) + b2
  
  # compute the Softmax class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss (cross-entropy loss)
    corect_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples
    #reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss # + reg_loss
    if i % 100 == 0:
        print ("i:", i, "loss: ", loss)
        ii.append(i)
        error.append(loss)
  
  # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples
  
  # apply the backpropate alghorithm(BP) the gradient 
    
  # first apply into parameters W2 and b2
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
  # next BP into hidden layer
    dhidden = np.dot(dscores, W2.T)
  # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
  # finally into W,b
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)
  
  # perform a parameter update
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2

# plot training loss

plt.figure(figsize=(10,8))
plt.plot(ii,error,label='error' )
plt.xlabel("iteration", fontsize=16)
plt.ylabel("error", fontsize=16)
plt.title("NN training process ", fontsize=18)
plt.legend()
plt.show()

# evaluate training set accuracy

hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy:::', (np.mean(predicted_class == y)))

h = 0.03
x_min, x_max = X[:, 0].min() - 0, X[:, 0].max() + 0
y_min, y_max = X[:, 1].min() - 0, X[:, 1].max() + 0
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8,8))
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

def test_prediction(XX):
    hidden_layer = np.maximum(0, np.dot(XX, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    prediction = np.argmax(scores, axis=1)
    return int(prediction)
    
XX = np.asarray([0.6,0.4])
print ("predicdted class for given point XX is :", test_prediction(XX))
XX = np.asarray([0.25,-0.75])
print ("predicdted class for given point XX is :", test_prediction(XX))