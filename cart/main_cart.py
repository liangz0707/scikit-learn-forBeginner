# coding:utf-8
__author__ = 'liangz14'

from numpy import *
import numpy as np

class treeNode():
    """
    分类回归树节点定义
    """
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left


def loadDataSet(fileName):
    """
    创建分类回归树的伪代码
    createTree：
        找到最佳的待切分特征：
            符合该节点不能再分，则这个节点就是叶子节点
            执行二元切分
            在右子树调用createTree()
            在左子树调用createTree()
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fitLine = map(float,curLine)
        dataMat.append(fitLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):

    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0, mat1

testMat = mat(eye(4))
a,b = binSplitDataSet(testMat,1,0.5)


