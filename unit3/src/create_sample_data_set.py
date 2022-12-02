from __future__ import division
import numpy as np
import future
import matplotlib.pyplot as plt


def create_sample_data_set(size, outer_width, outer_height, a, b, inner_width, inner_height):

    corners = np.array([[0, 0], [a, 0], [a+inner_width, 0],
                          [0, b], [a+inner_width, b],
                          [0, b+inner_height], [a, b+inner_height], [a+inner_width, b+inner_height]])
                          
    top_height = outer_height - (b + inner_height)
    right_width = outer_width - (a + inner_width)
    widths = np.array([a, inner_width, right_width, a, right_width, a, inner_width, right_width])
    heights = np.array([b, b, b, inner_height, inner_height, top_height, top_height, top_height])

    
    
    areas = widths * heights
    shapes = np.column_stack((widths, heights))

    regions = np.random.multinomial(size, areas/areas.sum())
    indices = np.repeat(range(8), regions)
    unit_coords = np.random.random(size=(size, 2))
    pts = unit_coords * shapes[indices] + corners[indices]

    #give the labels (supervise machine learnining) to the classes => 0 (robot free to go) or 1 (prohibited)
    
    y_out = np.ones(size, dtype='uint8')
    i = 0

    for pt in pts:
        
        if pt[0]>(5/7) or  pt[0]<(3/7) or pt[1]> (3/4) or pt[1]< (1/4):
            y_out[i] = 0
    
        i= i + 1

    return pts,y_out

plt.figure(figsize=(8,8))
X, y = create_sample_data_set(2000, 1, 1, 0.01, 0.01, 0, 0)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()