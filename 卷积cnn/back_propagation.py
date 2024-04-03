# -*- coding:utf-8 -*-
"""
日期：2021年08月12日
目的：反向传播
"""
import numpy as np
def logistic(x):
    return 1 / (1 + np.exp(-x))
def flip(arr):
    # 翻转函数
    arr = np.flipud(arr)
    arr = np.fliplr(arr)
    return arr
def pooling_drev(pooling_matrix_delta=[],pooling_padding=[],x_long=0,y_long=0,site=None,x_n=0,y_n=0,pooling_way=None,pooling_padding_num=0):

    pooling_drev = np.zeros_like(pooling_padding)
    # print(pooling_matrix_delta)
    m = 0

    if pooling_way=='max':
        for i in range(y_long):
            for j in range(x_long):
                pooling_drev[site[m][0] + y_n * i, site[m][1] + x_n * j] = pooling_matrix_delta[i, j]
                m += 1

    elif pooling_way == 'mean':
        for i in range(y_long):
            for j in range(x_long):
                pooling_drev[i * y_n:(i + 1) * y_n, j * x_n:(j + 1) * x_n] = pooling_matrix_delta[i, j] / (x_n * y_n)
                m += 1

    if pooling_padding_num!=0:

        pooling_drev=pooling_drev[int(pooling_padding_num/2):-int(pooling_padding_num/2),int(pooling_padding_num/2):-int(pooling_padding_num/2)]

    # print(pooling_drev)

    return pooling_drev

def activation_drev(activation_delta_matrix=[],activation_matrix=[],name=[],compare_num=[]):



    if name=='relu':
        relu_matrix = activation_matrix

        for x_relu in range(relu_matrix.shape[0]):
            for j_relu in range(relu_matrix.shape[1]):

                if relu_matrix[x_relu,j_relu]<=compare_num:
                    activation_delta_matrix[x_relu,j_relu]=0


    elif name == 'tanh':


        activation_matrix= 1.0 - np.tanh(activation_matrix) * np.tanh(activation_matrix)
        activation_delta_matrix = activation_delta_matrix / (activation_matrix+0.001)


    elif name == 'logic':
        activation_matrix=logistic(activation_matrix) * (1 - logistic(activation_matrix))
        activation_delta_matrix = activation_delta_matrix /(activation_matrix+0.001)

    elif name == 'sigmoid':
        pass
    elif name == 'leak_relu':
        pass


    return activation_delta_matrix

def cov_drev(cov_delta_matrix=[],matrix_padding=[],step=0,filter=[],site_cov=[],bias=0,learning_rate=10e-6):
    delta_cov=cov_delta_matrix
    delta_cov_matrix=np.zeros_like(matrix_padding)
    delta_cov_part=[]

    #delta矩阵（梯度）的更新
    # 首先卷积核要翻转过来
    # filter_flip=flip(filter)
    #不翻转也行(甚至效果更好？)
    filter_flip=filter
    # 先进行卷积



    #具体来说，我这种不是卷积，而是标记原来的位置，更具位置进行对应相乘，给矩阵中每个具体的输入元素分配对应的delta
    # 如果是卷积的话，那么应该先给featuremap转为一个适配input_matrix的格式,具体的方法为：卷积核的步长决定不同delta元素之间的补0数量，
    # delta之间补0数=step-1，比如卷积核每次移动两格，那么新的delta矩阵的delta元素之间相隔一个0，以此类推。外层补0数量等于卷积核大小减1，
    # 比如一个卷积核为3*3大小，那么外面就要补两个0
    # 举例：对一个6*6，并且补0一层变为8*8的input_map，在移动步长step为3的3*3卷积核卷积下形成2*2的feature_map,那么还原为反卷积的input_map
    # 的矩阵结构为：外面补2层0（3（卷积核filter大小为3）-1），里面每个delta元素间隔2个0（3（步长step为3）-1）
    # 这样里层（第一个和最后一个delta以内，反之）的补0数量等于（step-1）=2，再乘以补零次数（delta长-1）=1，这样2*1=2，即里面2个0，外层filter2*2+padding2=6个0,这样每行每列就有8个0，再加上2个delta元素
    # 反卷积中input_matrix是一个（8+2）*（8+2）=10*10的矩阵，与3*3的卷积核进行反卷积，刚好是一个卷积后的8*8的矩阵，即对应原来padding后的8*8input_matrix。
    # 当然也可以不考虑padding的0，这样算出来就是原input_matrix的图了

    #
    for i_delta_cov in range(delta_cov.shape[0]):
        for j_delta_cov in range(delta_cov.shape[1]):
            delta_cov_part.append((delta_cov[i_delta_cov, j_delta_cov])/(filter_flip))



    # 然后赋值

    for site_cov_counter in range(len(site_cov)):

        delta_cov_matrix[site_cov[site_cov_counter][0]:site_cov[site_cov_counter][0] + filter_flip.shape[0],
        site_cov[site_cov_counter][1]:site_cov[site_cov_counter][1] + filter_flip.shape[1]] += delta_cov_part[
            site_cov_counter]

        site_cov_counter += 1

    site_cov_counter=0
    updata_filter=np.zeros_like(filter)
    delta_cov_list=delta_cov.flatten()


    #卷积核的更新
    for i_site in range(len(site_cov)):


        pre_cov_matrix=matrix_padding[site_cov[i_site][0]:site_cov[i_site][0]+filter.shape[0],
        site_cov[i_site][1]:site_cov[i_site][1]+filter.shape[1]]

        # print(pre_cov_matrix)
        updata_filter+=delta_cov_list[i_site]/(pre_cov_matrix+0.005)
        site_cov_counter += 1


    filter+=updata_filter*learning_rate


    #偏置的更新

    bias+=np.sum(delta_cov)*learning_rate


    return delta_cov_matrix,filter,bias

def back_prop(delta=[],shape=[],cov_demo=[],learning_rate=10e-3):


    delta_matrix=delta.reshape(shape)
    filter_cov = []
    bias_cov = []

    #反向传播倒着来
    for i in range(len(cov_demo.Sequential)-1,-1,-1):


        if cov_demo.Sequential[i][0]=='pooling_add':

            dic={
            'pooling_matrix_delta' : delta_matrix,
            'pooling_padding' : cov_demo.Sequential[i][2],
            'x_long' : cov_demo.Sequential[i][3],
            'y_long' :cov_demo.Sequential[i][4],
            'site' : cov_demo.Sequential[i][5],
            'x_n':cov_demo.Sequential[i][6],
            'y_n' : cov_demo.Sequential[i][7],
            'pooling_way' : cov_demo.Sequential[i][8],
            'pooling_padding_num': cov_demo.Sequential[i][9]
            }


            pooling_drev_matrix=pooling_drev(**dic)
            delta_matrix=pooling_drev_matrix



        elif cov_demo.Sequential[i][0]=='act_add':


            dic={'activation_delta_matrix':delta_matrix,
                'activation_matrix':cov_demo.Sequential[i][1],
                 'name':cov_demo.Sequential[i][2],
                 'compare_num':cov_demo.Sequential[i][3]
            }

            delta_matrix=activation_drev(**dic)
            #
            # delta_matrix=np.zeros_like(delta_matrix)
            #
            # print(delta_matrix)


        elif cov_demo.Sequential[i][0]=='cov_add':
            # print(cov_demo.Sequential[i])


            dic = {'cov_delta_matrix': delta_matrix,
                   'matrix_padding': cov_demo.Sequential[i][2],
                   'step': cov_demo.Sequential[i][3],
                   'filter': cov_demo.Sequential[i][4],
                   # 'padding':cov_demo.Sequential[i][5],
                   # 'no_cov_matrix': cov_demo.Sequential[i][6],
                   'site_cov': cov_demo.Sequential[i][7],
                   'bias': cov_demo.Sequential[i][8],
                   'learning_rate':learning_rate
                   }

            delta_matrix,filter,bias=cov_drev(**dic)
            bias_cov.append(bias)
            filter_cov.append(filter)


        else:
            print('worry！!unexpectable layer!')


    filter_cov.reverse()
    bias_cov.reverse()

    return delta_matrix,filter_cov,bias_cov

