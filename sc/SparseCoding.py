# coding:utf-8
__author__ = 'liangz14'
#this is the demo from scikit-learn about : Incremental PCA
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
import math
import random as rd

from sklearn.decomposition import MiniBatchDictionaryLearning,DictionaryLearning
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA,SparseCoder

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
print len(X[1])
print type(X[1])
print len(X)
print type(X)

#字典学习部分
#dico = DictionaryLearning(n_components=100, alpha=1) #常用字典版本
dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500) #minibancth版本
V = dico.fit(X).components_

''''''
#字典显示部分
for i, comp in enumerate(V):#即会枚举索引，也会枚举值
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((5,5)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())


coder = SparseCoder(dictionary=V, transform_n_nonzero_coefs=13,
                            transform_alpha=None, transform_algorithm='omp')
x = coder.transform(X[1].reshape(1, -1))
print X[1]
x = np.ravel(np.dot(x, V))
print x

'''
#PCA学习部分
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(V) #选择对字典V用PCA 还是对原始数据X用PCA

#PCA显示部分
plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:,1], X_pca[:,0],c='r')
'''

plt.show()
