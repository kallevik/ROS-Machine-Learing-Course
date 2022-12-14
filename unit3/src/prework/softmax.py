import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


logits = [2.0, 0.0, 1.0]
print ("softmax function scores ::", softmax(logits))