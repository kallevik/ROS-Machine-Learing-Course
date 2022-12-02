import warnings
warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd


# load dataset
data = pd.read_csv('/home/user/catkin_ws/src/machine_learning_course/dataset/test_brakes.csv')

# remove Unmaned column
data.drop(['Unnamed: 0'], axis=1, inplace=True)

# print dataset
plt.figure(figsize=(10, 8))
plt.scatter(data['speed'],data['distance'])
plt.xlabel("% of max speed of axis 1", fontsize=16)
plt.ylabel("stop distance [deg]", fontsize=16)
plt.title("Dataset", fontsize=18)

plt.show()