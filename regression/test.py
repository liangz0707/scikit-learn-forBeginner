# coding:utf-8
__author__ = 'liangz14'

"""
    这部分主要用来测试矩阵计算的知识
"""

import numpy as np
import numpy.random as rd


def show_solve_matrix1():
    """
        使用解析解直接求解 Ax=B的方程
        首先生成A、x 然后计算一个B
        然后利用A B 求解除x
        符合回归问题： 已知A、B  求x: A->B
    :return:
    """
    # 得到变量Ax=B
    A = rd.dirichlet([1, 2, 3], (3,))  # 生成一个矩阵，不用关心具体意义
    x = np.asarray([2, 3, 5])
    B = np.dot(A, x)
    # 这里有了 A 和 B 尝试求解 x

    # 由于不知道A是否可逆，所以使用A的伪逆
    Ap = np.linalg.pinv(A)
    # 得到解析解
    print np.dot(Ap, B)
    egen_val = np.linalg.eigvals(A)


def show_solve_matrix2():
    """
        这个函数不同之处在于A不在是方阵：n*m
        这里我们令n>m
        由于n>m所以不可能满秩，所以很有可能没有解（下面的例子是有解的情况）
        如果没有解，那么求出的x 会让 Ax最接近B ，但是不会相等（因为无解）
        x 是不是矩阵结果都是一个意思的
    :return:
    """
    # 得到变量Ax=B
    A = rd.dirichlet([1, 2, 3], (5,))  # 5 × 3
    print A
    x = np.asarray([[2, 3], [1, 3], [1, -1]])
    #x = np.asarray([2, 3, 1])
    B = np.dot(A, x)
    # 这里有了 A 和 B 尝试求解 x

    # 由于不知道A是否可逆，所以使用A的伪逆
    Ap = np.linalg.pinv(A)

    # 得到解析解
    print np.dot(Ap, B)


def show_solve_matrix3():
    """
        这个函数不同之处在于A不在是方阵：n*m
        这里我们令n<m
        由于n<m所以:1.可能不满秩、2.满秩
        所以：1.可能有很多解，2.可能无解，3.可能一个解
        那么：
        如果很多解，那么求出的解一定让x的二范式最小
        如果没有解，那么就是最接近；
        如果唯一解，那就是唯一的解；

        我给出的例子是有很多组解的，所以得到的解和我给的x不同，而是让x二范式最小的那一组
        x 是不是矩阵结果都是一个意思的
    :return:
    """
    # 得到变量Ax=B
    A = rd.dirichlet([1, 2, 3, 3, 3], (3,))  # 5 × 3
    print A
    x = np.asarray([2, 3, 5, 2, 3])
    B = np.dot(A, x)
    # 这里有了 A 和 B 尝试求解 x

    # 由于不知道A是否可逆，所以使用A的伪逆
    Ap = np.linalg.pinv(A)

    # 得到解析解
    print np.dot(Ap, B)


# ===================== 以下三个内容，都是针对方程求解的，回归的过程时A->B,x为映射的矩阵（向量）==========
show_solve_matrix1()
show_solve_matrix2()
show_solve_matrix3()
# ===================== 以下内容，都是针对方程求解的，回归的过程时A->B,x为映射的矩阵（向量）==========
