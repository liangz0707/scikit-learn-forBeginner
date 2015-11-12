# coding:utf-8
__author__ = 'liangz14'
#this is the demo from scikit-learn about : Incremental PCA
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
import math
import random as rd

from sklearn.datasets import load_iris

def get_patch(filename,patch_size = 5,patch_num = 1000):
    src = io.imread(filename)
    im = color.rgb2lab(src[:,:,0:3])[:,:,0]
    patchs = []
    for i in range(patch_num):
        y,x = [rd.randint(0,im.shape[d]-patch_size) for d in [0,1]]
        tmp = im[y:y+patch_size, x:x+patch_size].reshape((patch_size*patch_size))
        tmp = tmp - np.mean(tmp)
        patchs.append(tmp)
    return patchs


n_components = 2

X = get_patch('T.png')
print len(X)
print type(X)


#plt.figure(figsize=(8, 8))
#plt.scatter(X_pca[:,1], X_pca[:,0],c='r')

#plt.axis([-4, 4, -1.5, 1.5])
#plt.show()
