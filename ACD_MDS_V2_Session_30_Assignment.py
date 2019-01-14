# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 22:19:49 2019

@author: Eliud Lelerai
"""

from skimage import io
from sklearn.cluster import KMeans
import numpy as np

import scipy.misc
face = scipy.misc.face()


import matplotlib.pyplot as plt
plt.gray()
plt.imshow(face)
plt.show()

from scipy.misc import imsave
imsave('raccoon.png', face)

from skimage import io
from sklearn.cluster import KMeans
import numpy as np
 
image = io.imread('raccoon.png')
io.imshow(image)
io.show()
 
rows = image.shape[0]
cols = image.shape[1]
  
image = image.reshape(image.shape[0]*image.shape[1],3)
kmeans = KMeans(n_clusters = 5, n_init=10, max_iter=200)
kmeans.fit(image)
 
clusters = np.asarray(kmeans.cluster_centers_,dtype=np.uint8) 
labels = np.asarray(kmeans.labels_,dtype=np.uint8 )  
labels = labels.reshape(rows,cols); 
 
np.save('codebook_racoon.npy',clusters)    
io.imsave('compressed_raccoon.png',labels);