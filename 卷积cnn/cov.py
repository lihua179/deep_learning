# -*- coding:utf-8 -*-
"""
日期：2021年08月12日
目的：
"""
import numpy as np


def cov_2D(matrix_one, step=2, filter=[], padding=2, bias=0):
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
            site_cov_.append((i, j))

    # print(feature_matrix)
    # print(bias)
    sum_filter = feature_matrix + bias
    # print(matrix_one)

    return sum_filter, matrix_one, site_cov_


def relu(feature_map, compare_num=0):
    # print('激活层')
    relu_out = np.zeros_like(feature_map)
    # print(relu_out)

    #
    relu_out[:] = np.maximum(feature_map[:], compare_num)
    # relu_out[:] = np.maximum(tanh(feature_map[:]), compare_num)
    # print('relu_out',relu_out)
    #
    return relu_out


def tanh(feature_map):
    return np.tanh(feature_map)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def padding_(feature_matrix, padding):
    zeros_matrix = np.zeros([feature_matrix.shape[0] + padding, feature_matrix.shape[1] + padding])

    semi_num = int(padding / 2)
    zeros_matrix[semi_num:-semi_num, semi_num:-semi_num] = feature_matrix[:, :]

    feature_matrix = zeros_matrix

    return feature_matrix


def pooling_layer_2D(feature_matrix, x_n=2, y_n=2, pooling_way='max', padding=2):
    # 参数：
    # 特征矩阵，水平方向每格长度，竖直方向每格长度，池化方式（最大池化/均值池化），是否补0，补几层0

    if padding != 0:
        feature_matrix = padding_(feature_matrix, padding)

    elif padding == 0:
        feature_matrix = feature_matrix

    x_step = feature_matrix.shape[1]  # 列，水平方向
    y_step = feature_matrix.shape[0]  # 行，竖直方向

    # 水平方向移动次数
    x_long = int(x_step / x_n)

    # 竖直方向移动次数
    y_long = int(y_step / y_n)

    pooling_matrix = np.zeros([y_long, x_long])
    site = []

    for i in range(y_long):
        # 行作为外层
        for j in range(x_long):
            # 列作为内层

            if pooling_way == 'max':
                a = feature_matrix[i * y_n:(i + 1) * y_n, j * x_n:(j + 1) * x_n]

                pooling_matrix[i, j] = np.max(a)

                site.append(np.unravel_index(np.argmax(a), a.shape))




            elif pooling_way == 'mean':
                pooling_matrix[i, j] = np.mean(feature_matrix[i * y_n:(i + 1) * y_n, j * x_n:(j + 1) * x_n])

    # print(site)

    return pooling_matrix, x_long, y_long, site, feature_matrix


class cov_2D_list:

    def __init__(self, matrix=[]):
        self.matrix = matrix
        self.matrix_padding = None
        self.step = None
        self.filter = None
        self.padding = None
        self.activate_fun = None
        self.pooling_padding = None
        self.x_long = None
        self.y_long = None
        self.site = None
        self.site_cov = None
        self.bias = None

        pass

    def cov_2D_pre(self, step=1, filter=filter, padding=0, bias=0):

        self.step = step
        self.filter = filter
        self.padding = padding
        self.bias = bias
        self.matrix, self.matrix_padding, self.site_cov = cov_2D(matrix_one=self.matrix, step=self.step,
                                                                 filter=self.filter, padding=self.padding,
                                                                 bias=self.bias)

    def act_fun(self, activate_function=None, compare_num=0):
        if activate_function == 'relu':
            self.matrix = relu(self.matrix, compare_num)
        if activate_function == 'tanh':
            self.matrix = tanh(self.matrix)
        if activate_function == 'logic':
            self.matrix = logistic(self.matrix)

        # else:

    def pooling(self, x_n=2, y_n=2, pooling_way='max', padding=2):

        if pooling_way == 'max':

            self.matrix, self.x_long, self.y_long, self.site, self.pooling_padding = pooling_layer_2D(self.matrix,
                                                                                                      x_n=x_n, y_n=y_n,
                                                                                                      pooling_way='max',
                                                                                                      padding=padding)
        elif pooling_way == 'mean':

            self.matrix, self.x_long, self.y_long, self.site, self.pooling_padding = pooling_layer_2D(self.matrix,
                                                                                                      x_n=x_n, y_n=y_n,
                                                                                                      pooling_way='mean',
                                                                                                      padding=padding)


class nn:
    def __init__(self, matrix=[]):
        self.cov_2D = cov_2D_list(matrix)
        self.matrix = matrix
        self.act_fun = None
        self.Sequential = []
        self.matrix_shape = []
        self.Sequential_dic = {'cov_add': ['self.cov_2D.matrix:卷积后的特征矩阵',
                                           'self.cov_2D.matrix_padding:卷积前输入图像的预先补0扩张后的矩阵',
                                           'step：卷积移动步长', 'filter：卷积核', 'padding：补0的数量',
                                           'no_cov_matrix:卷积前的输入矩阵',
                                           'self.cov_2D.site_cov:卷积过程中产生的映射前原来输入元素的坐标',
                                           'bias:特特征图偏置'],

                               'act_add': ['self.cov_2D.matrix:激活后的特征矩阵', 'activate_function:激活函数名',
                                           'compare_num：如果是relu函数，那么就是比较值得阈值，否则为空'],

                               'pooling_add': ['self.cov_2D.matrix:池化后的特征矩阵',
                                               'self.cov_2D.pooling_padding:池化前补0后的矩阵',
                                               'self.cov_2D.x_long：池化块水平方向移动次数',
                                               'self.cov_2D.y_long：垂直方向移动次数',
                                               'self.cov_2D.site：池化块原来最大值的相对位置', 'x_n：池化块的长',
                                               'y_n：池化块的高', 'pooling_way：池化方式', 'pooling_padding_num:补数量']
                               }

    def cov_add(self, step=None, filter=None, padding=None, bias=None):
        no_cov_matrix = self.cov_2D.matrix
        self.cov_2D.cov_2D_pre(step=step, filter=filter, padding=padding, bias=bias)
        self.Sequential.append(
            ['cov_add', self.cov_2D.matrix, self.cov_2D.matrix_padding, step, filter, padding, no_cov_matrix,
             self.cov_2D.site_cov, bias])
        # 获得相应的参数：卷积后的矩阵,卷积前但padding后的矩阵,step,filter,padding
        # self.Sequential_dic.append(['cov_add',self.cov_2D.matrix,self.cov_2D.matrix_padding,step,filter,padding])

    def act_add(self, activate_function=None, compare_num=None):
        self.cov_2D.act_fun(activate_function=activate_function, compare_num=compare_num)
        self.Sequential.append(['act_add', self.cov_2D.matrix, activate_function, compare_num])
        # self.Sequential_dic.append(['act_add',self.cov_2D.matrix,activate_function,compare_num])

    def pooling_add(self, x_n=None, y_n=None, pooling_way=None, pooling_padding_num=None):
        # no_padding_matrix=self.cov_2D.matrix
        self.cov_2D.pooling(x_n=x_n, y_n=y_n, pooling_way=pooling_way, padding=pooling_padding_num)
        self.Sequential.append(
            ['pooling_add', self.cov_2D.matrix, self.cov_2D.pooling_padding, self.cov_2D.x_long, self.cov_2D.y_long,
             self.cov_2D.site, x_n, y_n, pooling_way, pooling_padding_num])
        # self.Sequential_dic.append(['pooling_add',self.cov_2D.matrix,self.cov_2D.x_long,self.cov_2D.y_long,self.cov_2D.site,x_n, y_n,pooling_way,pooling_padding])

    def flatten(self):
        self.matrix_shape = self.cov_2D.matrix.shape
        self.cov_2D.matrix = self.cov_2D.matrix.flatten()
