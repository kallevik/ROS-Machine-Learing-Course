import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#data set
points = np.array([(1, 1), (2, 4), (3, 1), (9, 3)])
x = points[:,0]
y = points[:,1]
x=np.array([0,1,2,3,4,5,6,7,8,9])
y=np.array([3,0.8,0.9,1.5,-0.8,-1,5,-5,0.1,-5])

# calculate polynomial for the simple data set
order1=np.polyfit(x,y,1)
order2=np.polyfit(x,y,2)
order5=np.polyfit(x,y,5)
order9=np.polyfit(x,y,9)

xp=np.linspace (-2,10,100)
  
plt.figure(figsize=(16, 8))
plt.plot(xp,np.polyval(order1,xp),'-g', label = 'order 1')
plt.plot(xp,np.polyval(order2,xp),'b--', label = 'order 2')
plt.plot(xp,np.polyval(order5,xp),'r', label = 'order 5')
plt.plot(xp,np.polyval(order9,xp), '--', label = 'order 9')
plt.plot(x,y,'o', 'r')
plt.ylim(-15, 15)
plt.xlabel("x", fontsize=16)
plt.ylabel("y", fontsize=16)
plt.title ("Regression (polynomials)", fontsize=18)
plt.legend()
plt.show()

fittedParameters = np.polyfit(x, y, 3)
print('Single value prediction:', np.polyval(fittedParameters, 2.3))