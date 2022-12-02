# Load weights
import numpy as np

def load_weights():
    
    W2 = np.load('/home/user/catkin_ws/src/machine_learing_course/weights/NN_trainer/w2.npy')
    b2 = np.load('/home/user/catkin_ws/src//machine_learing_course/weights/NN_trainer/b2.npy')
    W = np.load('/home/user/catkin_ws/src//machine_learing_course/weights/NN_trainer/w.npy')
    b = np.load('/home/user/catkin_ws/src//machine_learing_course/weights/NN_trainer/b.npy')
    
    return W2,b2,W,b


W2,b2,W,b = load_weights()

# Predict
def test_prediction(XX):
    hidden_layer = np.maximum(0, np.dot(XX, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    prediction = np.argmax(scores, axis=1)
    return int(prediction)
    
XX = np.asarray([0.6,0.4])
print("predicdted class for given point XX is :", test_prediction(XX))
XX = np.asarray([0.25,0.6])
print("predicdted class for given point XX is :", test_prediction(XX))