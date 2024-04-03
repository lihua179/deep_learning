# -*- coding:utf-8 -*-
"""
日期：2021年08月12日
目的：
"""
import cov
import back_propagation as bp
import numpy as np


#
#
# cov_demo=cov.nn(matrix)
# #建立第一层卷积层
# cov_demo.cov_add(step=1, filter=filter[0], padding=0,bias=bias[0])
# cov_demo.act_add(activate_function='relu',compare_num=0.5)
# cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='max', pooling_padding_num=2)
# #建立第二层卷积层
# cov_demo.cov_add(step=1, filter=filter[1], padding=0,bias=bias[1])
# cov_demo.act_add(activate_function='relu',compare_num=0.5)
# cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='mean', pooling_padding_num=2)
# cov_demo.flatten()


# print('模拟全连接层输入')
# print(cov_demo.cov_2D.matrix)
# print('cov_demo.matrix_shape')
# print(cov_demo.matrix_shape)
# delta=cov_demo.cov_2D.matrix

# delta_matrix,filter,bias=bp.back_prop(delta=delta,shape=cov_demo.matrix_shape,cov_demo=cov_demo)

def cov_programe(matrix=[], filter=[], bias=[], learning_rate=10e-3):
    cov_demo = cov.nn(matrix)
    # 建立第一层卷积层
    cov_demo.cov_add(step=1, filter=filter[0], padding=0, bias=bias[0])
    cov_demo.act_add(activate_function='tanh', compare_num=0)
    cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='max', pooling_padding_num=2)
    # # 建立第二层卷积层
    # cov_demo.cov_add(step=1, filter=filter[1], padding=2, bias=bias[1])
    # cov_demo.act_add(activate_function='relu', compare_num=0)
    # cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='mean', pooling_padding_num=2)
    # # 建立第三层卷积层
    # cov_demo.cov_add(step=1, filter=filter[1], padding=2, bias=bias[1])
    # cov_demo.act_add(activate_function='tanh', compare_num=0)
    # cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='mean', pooling_padding_num=2)
    cov_demo.flatten()

    # 模拟全连接层梯度传递
    matrix = np.random.random(cov_demo.matrix_shape)
    # print(cov_demo.matrix_shape)
    matrix = matrix.flatten()

    delta = (matrix * np.random.standard_normal(size=len(matrix)) * 0.5) * 0.1
    print('delta')
    print(delta)
    # print(delta.shape)
    # print(cov_demo.Sequential_dic)
    # print('delta')
    # print(delta)

    delta_matrix, filter, bias = bp.back_prop(delta=delta, shape=cov_demo.matrix_shape, cov_demo=cov_demo,
                                              learning_rate=learning_rate)

    return filter, bias


# 数据的输入
# matrix=np.random.random([10,10])
filter = np.random.random([2, 2, 4])
bias = np.random.random([4])
# print('filter')
# print(filter[:2])
# print('bias')
# print(bias[:2])
# print()
for i in range(2):
    matrix = np.random.random([6, 6])
    filter, bias = cov_programe(matrix=matrix, filter=filter, bias=bias, learning_rate=1e-2)
    print(i)
    # print('matrix')
    # print(matrix)

    print('filter')
    print(filter[0])
    print('bias')
    print(bias[0])
    print()

# print(filter, bias)

# print('delta_matrix')
# print(delta_matrix)
# print('filter')
# print(filter)
# print('bias')
# print(bias)