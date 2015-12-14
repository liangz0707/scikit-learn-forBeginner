# coding:utf-8
__author__ = 'liangz14'
'''
    knn算法源代码实现：
'''
#导入模块
import numpy as np
from numpy import *
import operator
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
import math
import random as rd

#读取训练文件
def createDataSet(file_name):
    #输入
    source = list()
    #标签
    target = list()
    with open(file_name) as t_file:
        #所有行
        lines = t_file.readlines()
        #对于没一行吧数据和标签分离，分别保存在source和target中
        for i,line in enumerate(lines[1:]):
            tmp_str = line.split(",")   #通过，分割。csv的格式：target,i1,i2,i3,i4,……
            target.append(int(tmp_str[0]))
            source.append([int(c) for c in tmp_str[1:]])

    return target, source
#读取训练数据
group,labels= createDataSet("train.csv")

'''
    kNN 算法步骤：
    1.计算一直类别数据集中和当前点的距离
    2.按照距离递增排序
    3.选取与当前点距离最近的k个点
    4.确定出现的频率
    5.返回出现频率最高的点
'''
#进行测试，输入分别为测试数据、训练输入、训练标签、k值
def classify0(inx ,dataSet ,labels,k):
    diff = tile(inx,(len(dataSet),1)) -dataSet   #tile是将向量inx复制多列构成[inx,int,……,inx]维数和dataSet相同以便求和每一个的距离
    diffMat = diff**2   #计算差值的平方
    sqSpace = diffMat.sum(axis=1)  #求平方和
    distance = sqSpace**0.5  #开根号计算实际距离
    sortedDistance = distance.argsort() #argsort的作用是返回排序后的索引，但是distance并未排序

    classCount={}
    for i in range(k):
        c = labels[sortedDistance[k]]  #读取前k个输入的标签
        classCount[c] = classCount.get(c,0) +1.0/distance[k]  #统计每个标签出现次数
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)#对出现次数进行排序
    return sortedClassCount[0][0]#返回出现次数最多的标签

#测试
print group[3]
print classify0(labels[3],labels[0:10000],group[0:10000],5)
