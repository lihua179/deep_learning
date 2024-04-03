# -*- coding:utf-8 -*-
"""
日期：2021年08月21日
目的：
"""
import numpy as np

def activate(x):
    return np.tanh(x)

def derive_act(x):
    return np.tanh(x)*(1-np.tanh(x))

def new_x_input_fun(h0, x_input1):
    # 将上一个计算激活后的h0和当前新的x_input进行合并
    new_x_input = np.hstack((h0, x_input1))
    return new_x_input


def new_x_w(w_hh, w_x):
    # 将上一个计算激活后的h0和当前新的x_input进行合并
    new_x_w = np.hstack((w_hh, w_x))
    return new_x_w


def new_x_h_fun(new_x_input, new_x_w):
    # 将合并后的输入和矩阵进行矩阵乘法
    new_h = activate(np.dot(new_x_input, new_x_w.T))
    return new_h


def wpt_pre(new_xh_w, layer, m):
    weight_middle = []
    weight_middle.append(new_xh_w.T)
    # 第二个矩阵的行就是合并后的矩阵的行
    weight_middle.append(np.random.random([new_xh_w.shape[0], layer[0]]))
    for i in range(1, len(layer)):
        weight_middle.append(np.random.random([layer[i - 1], layer[i]]))

    # 最终的列就是最终输出的长度
    weight_middle.append(np.random.random([layer[-1], m]))

    return weight_middle

def back_prop(weight_middle=[],y_new_x_input_layer=[],new_x_input_layer_act=[],learning_rate=0.01,x_input=[]):
    delta_list = []

    # 误差模拟
    delta_last = np.random.random(m) * 100
    delta_list.append(delta_last)
    count=1
    # print()

    for i in range(len(x_input)):

        for j in range(len(weight_middle)-1,0,-1):

            delta_list.append(np.dot(delta_list[-1]*derive_act(y_new_x_input_layer[-count]), weight_middle[j].T))
            count+=1

        #之前矩阵不是合并了吗，现在把这份delta分开算，delta_left是上一个输出的误差，而上一个输出的误差会继续进行梯度传递

        delta_left=np.dot(delta_list[-1]*derive_act(y_new_x_input_layer[-count]), weight_middle[0].T)[:m]
        count += 1

        delta_list.append(delta_left)

    delta_list.reverse()

    #更新权重

    count=1


    #第一次赋值权重
    for i in range(0, len(weight_middle)):
        # print(i)

        weight_middle[i] += learning_rate * np.dot(np.array([delta_list[count]]).T,
                                                   [new_x_input_layer_act[count - 1]]).T

        count += 1

    #第二次开始赋值权重
    for j in range(1,len(x_input)):


        new_x_input_ = new_x_input_fun(new_x_input_layer_act[count - 1], x_input[j])



        weight_middle[0] += learning_rate * np.dot(np.array([delta_list[count]]).T,
                                                   [new_x_input_]).T

        count += 1



        for i in range(1,len(weight_middle)):

            weight_middle[i] += learning_rate *np.dot(np.array([delta_list[count]]).T,[new_x_input_layer_act[count]]).T

            count+=1


    return weight_middle
#隐藏神经层
layer=[2,3]

#输入（样本数，每次样本输入的次数和长度）
# 输入模拟
# 比如一共5段序列作为样本（每段长度为16），每段序列分2小批送进去，每小批的长度为8
x_input_list =np.random.random([5,2,8])

# m: 最终输出的维度
m=1

# n: 权重的输出有几维度
n=2



#学习率
learning_rate=0.05
def rnn_forward_net(layer=[], x_input_list=np.random.random([10,10,4]), m=0, n=0,learning_rate=0.01):
    # 初始化
    x_input=x_input_list[0]
    print(x_input)
    # n :权重的输出有几维度
    w_x = np.random.random([n, x_input.shape[1]])

    # m: 最终输出的维度
    w_hh = np.random.random([n, m])

    # 连结后新的权重
    new_xh_w = new_x_w(w_hh, w_x)

    # 第一个最终输出的维度的初始化
    new_x_h = np.zeros(w_hh.shape[1])

    # 中间的隐藏层初始化
    weight_middle = wpt_pre(new_xh_w, layer, m)
    y_new_x_input_layer = []
    new_x_input_layer_act = []






    for k in range(len(x_input_list)):

        x_input=x_input_list[k]
        for j in range(x_input.shape[0]):
            # print(x_input.shape[0])

            # print(x_input.shape[0])
            # 将上一个输出和这次输入进行合并
            new_x_input_ = new_x_input_fun(new_x_h, x_input[j])

            # 储存第一次的激活输出（即普通输入）
            new_x_input_layer_act.append(new_x_input_)

            # # 储存第一次的普通输出（即普通输入）
            # y_new_x_input_layer_act.append(new_x_input_)

            for i in range(len(weight_middle)):
                # 将上一个的激活输出当做输入
                y_new_x_input_layer.append(np.dot(new_x_input_layer_act[-1], weight_middle[i]))

                # 这次的输出进行激活
                new_x_input_layer_act.append(activate(y_new_x_input_layer[-1]))



            # 将上一个的激活输出当做输入
            # y_new_x_input_last = np.dot(new_x_input_layer_act[-1], weight_middle[-1])

            # 这次的输出进行激活
            # new_x_input_last_act = activate(y_new_x_input_last)

            new_x_h = new_x_input_layer_act[-1]

        weight_middle = back_prop(weight_middle=weight_middle,y_new_x_input_layer=y_new_x_input_layer,new_x_input_layer_act=new_x_input_layer_act,learning_rate=learning_rate,x_input=x_input)

        # print(weight_middle)


    return weight_middle


weight_middle= rnn_forward_net(layer=layer, x_input_list=x_input_list, m=m, n=n,learning_rate=learning_rate)








