# # -*- coding:utf-8 -*-
# """
# 日期：2021年08月12日
# 目的：
# """
# import cov
#
# import numpy as np
#
# def cov_programe(matrix=[], filter=[], bias=[], learning_rate=10e-3):
#     cov_demo = cov.nn(matrix)
#     # 建立第一层卷积层
#     cov_demo.cov_add(step=1, filter=filter[0], padding=0, bias=bias[0])
#     cov_demo.act_add(activate_function='tanh', compare_num=0)
#     cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='mean', pooling_padding_num=2)
#     # 建立第二层卷积层
#     cov_demo.cov_add(step=1, filter=filter[1], padding=2, bias=bias[1])
#     cov_demo.act_add(activate_function='relu', compare_num=0)
#     cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='mean', pooling_padding_num=2)
#     # 建立第三层卷积层
#     cov_demo.cov_add(step=1, filter=filter[2], padding=2, bias=bias[2])
#     cov_demo.act_add(activate_function='tanh', compare_num=0)
#     cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='max', pooling_padding_num=2)
#     cov_demo.flatten()
#
#     return cov_demo.matrix_shape ,cov_demo.cov_2D.matrix
#
#
# filter = np.random.random([8,4,4])
# bias = np.random.random([4])
#
# x_train=np.load('x_train.npy')
# y_train=np.load('y_train.npy')
#
# x_test=np.load('x_test.npy')
# y_test=np.load('y_test.npy')
#
#
# print(x_train.shape)
# # print(len(x_train[0]))
# print(y_train.shape)
#
# # print()
# # print(x_test[0])
# print(y_test[0])
# y=np.zeros([10])
# print(y)
# print()
# y[y_test[0]]=1
# print(y)
#
# for i in range(1):
#
#     matrix=x_train[0]
#     flatten_len ,cov_demo_cov_2D_matrix= cov_programe(matrix=matrix, filter=filter, bias=bias, learning_rate=1e-2)
#     print(flatten_len,cov_demo_cov_2D_matrix.shape)
#
import numpy as np
pre_num = np.random.randint(0,60000)
print(pre_num)