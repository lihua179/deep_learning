# -*- coding:utf-8 -*-
"""
日期：2021年08月16日
目的：
"""
import numpy as np
import time as t
def cov_2D(matrix_one, step=2, filter=[], padding=2,bias=0):
    # 输入输入数据，卷积核每次移动步数量，卷积核（任意形状），补0数量，返回原始图和特征图

    # 0矩阵初始化
    zeros_matrix = np.zeros([matrix_one.shape[0] + padding, matrix_one.shape[1] + padding])

    # 平分0
    semi_num = int(padding / 2)

    # 把原始图放入零矩阵中
    if semi_num != 0:
        zeros_matrix[semi_num:-semi_num, semi_num:-semi_num] = matrix_one[:, :]

        # 补0后的原始图像
        matrix_one = zeros_matrix

    # print(matrix_one)
    # print(semi_num)

    # 根据这个原始图像判断卷积核

    # 在水平方向能移动的步伐(列数)
    step_can_move_x = (matrix_one.shape[1] - filter.shape[1]) // step

    # print('水平方向剩余格数', matrix_one.shape[2] - filter.shape[2], '可移动格数', step_can_move_x)
    # 垂直方向可移动步长(行数)
    step_can_move_y = (matrix_one.shape[0] - filter.shape[0]) // step
    # print('垂直方向剩余格数', matrix_one.shape[1] - filter.shape[1], '可移动格数', step_can_move_y)

    # 特征矩阵的初始化(可移动格数加自身占据一格)
    feature_matrix = np.zeros([step_can_move_y + 1, step_can_move_x + 1])

    # 偏移量b
    # b=np.random.random()

    # n, m = 0, 0
    site_cov_ = []

    for i in range(0, matrix_one.shape[0] - filter.shape[0] + 1, step):
        # 先确定每一行
        # print('i:', i, matrix_one.shape[1] - filter.shape[1] + 1, step)

        for j in range(0, matrix_one.shape[1] - filter.shape[1] + 1, step):
            # print(round(i / step,2)[2])
            # 再确定每一列

            feature_matrix[int(i / step), int(j / step)] = np.sum((matrix_one[i:i + filter.shape[0],
                                                                   j:j + filter.shape[1]] * filter[:, :]))
            site_cov_.append((i,j))


    # print(feature_matrix)
    # print(bias)
    sum_filter = feature_matrix+bias

    return sum_filter, matrix_one, site_cov_
matrix=np.random.randint(-1,5,[600,600])
print('input_matrix')
print(matrix)
filter=np.random.randint(-1,5,[20,20])
print('filter')
print(filter)
padding=0
feature_map, matrix_one, site_cov_=cov_2D(matrix, step=2, filter=filter, padding=padding,bias=0)
# print('matrix_one')
# print(matrix_one)

print('feature_map')
print(feature_map)

padding_outter=(filter.shape[0]-1)*2#外层补0数量等于（卷积核长-1）*2

step=2
padding_inner=(step-1)*(feature_map.shape[0]-1)#内层补0数量等于（step-1）*(delta长-1）
filter_shape=filter.shape[0]
feature_map_shape=feature_map.shape[0]
feature_map_delta=np.ones_like(feature_map)*np.random.random(feature_map.shape)
def cov_input_map_comput(filter_shape=[],feature_map_shape=[],step=1,feature_map=[]):


    padding_outter = (filter_shape - 1) * 2  # 外层补0数量等于（卷积核长-1）*2
    padding_inner = (step - 1) * (feature_map_shape - 1)  # 内层补0数量等于（step-1）*(delta长-1）
    long = padding_outter + padding_inner + feature_map_shape#外层补0数+内层补0数+delta数
    cov_input_map = np.zeros([long, long])

    for i in range(feature_map.shape[0]):
        for j in range(feature_map.shape[1]):
            y_site = int(padding_outter / 2 + i * step)
            x_site = int(padding_outter / 2 + j * step)
            cov_input_map[y_site, x_site] = feature_map[i, j]

    return cov_input_map

cov_input_map=cov_input_map_comput(filter_shape=filter_shape,feature_map_shape=feature_map_shape,step=step,feature_map=feature_map_delta)
print('cov_input_map')
print(cov_input_map)


filter_flip=np.flip(filter)
t1=t.time()
delta_feature_map, matrix_one_2, site_cov_2=cov_2D(cov_input_map, step=1, filter=filter_flip, padding=0,bias=0)
print('时间差(业界卷积法)')
print(t.time()-t1)

print()
def cov_drev(cov_delta_matrix=[],matrix_padding=[],filter=[],site_cov=[]):
    delta_cov=cov_delta_matrix
    delta_cov_matrix=np.zeros_like(matrix_padding)
    delta_cov_part=[]

    #delta矩阵（梯度）的更新
    filter_flip=filter
    # 先进行卷积
    for i_delta_cov in range(delta_cov.shape[0]):
        for j_delta_cov in range(delta_cov.shape[1]):
            delta_cov_part.append(filter_flip * delta_cov[i_delta_cov, j_delta_cov])

    # print('delta_cov_part')
    # print(delta_cov_part)
    # delta_cov_part=np.array(delta_cov_part)


    # print()

    # 然后赋值
    # delta_cov_matrix=delta_cov_matrix.tolist()
    # delta_cov_part=delta_cov_part.tolist()
    # delta_cov_matrix=float(delta_cov_matrix)
    # print(delta_cov_matrix)
    delta_cov_matrix=delta_cov_matrix*0.0

    for site_cov_counter in range(len(site_cov)):
        # print(delta_cov_matrix[site_cov[site_cov_counter][0]:site_cov[site_cov_counter][0] + filter_flip.shape[0],
        # site_cov[site_cov_counter][1]:site_cov[site_cov_counter][1] + filter_flip.shape[1]])
        # print(delta_cov_part[
        #     site_cov_counter])
        # print()
        # print(type(delta_cov_matrix[site_cov[site_cov_counter][0]:site_cov[site_cov_counter][0] + filter_flip.shape[0],
        # site_cov[site_cov_counter][1]:site_cov[site_cov_counter][1] + filter_flip.shape[1]]))
        # print(type(delta_cov_part[
        #     site_cov_counter]))

        delta_cov_matrix[site_cov[site_cov_counter][0]:site_cov[site_cov_counter][0] + filter_flip.shape[0],
        site_cov[site_cov_counter][1]:site_cov[site_cov_counter][1] + filter_flip.shape[1]] += delta_cov_part[
            site_cov_counter]

        site_cov_counter += 1



    return delta_cov_matrix
t2=t.time()
delta_cov_matrix=cov_drev(cov_delta_matrix=feature_map_delta,matrix_padding=matrix_one,filter=filter,site_cov=site_cov_)
print('时间差(对照相乘法)')
print(t.time()-t2)
print('卷积delta_cov_matrix')
print(delta_feature_map)
print('非卷积delta_cov_matrix')
print(delta_cov_matrix)
