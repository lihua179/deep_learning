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


# def h0_compu(x_input_now,w_x):
#     #输出激活后的上一个h0
#     h0 = activate(np.dot(x_input_now,w_x.T))
#     return h0

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


def new_x_h_w_fun(new_x_h, w_hhh, w_hhh_last):
    new_x_h_whhh = np.dot(new_x_h, w_hhh)
    new_x_h_whhh_last = np.dot(new_x_h_whhh, w_hhh_last)

    return new_x_h_whhh_last


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


x_input = np.random.random([2, 3])


def rnn_forward_net(layer=[2, 3], x_input=np.random.random([2, 3]), m=4, n=2):
    # 初始化

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
    for j in range(x_input.shape[0]):
        # print(x_input.shape[0])

        # print(x_input.shape[0])
        # 将上一个输出和这次输入进行合并
        new_x_input_ = new_x_input_fun(new_x_h, x_input[j])

        # 储存第一次的激活输出（即普通输入）
        new_x_input_layer_act.append(new_x_input_)

        # # 储存第一次的普通输出（即普通输入）
        # y_new_x_input_layer_act.append(new_x_input_)

        for i in range(len(weight_middle)-1):
            #这里减一是因为最后一次可以用其他函数激活（比如交叉熵，而且可以直接为后面的反向传播使用）


            # 将上一个的激活输出当做输入
            y_new_x_input_layer.append(np.dot(new_x_input_layer_act[-1], weight_middle[i]))

            # 这次的输出进行激活
            new_x_input_layer_act.append(activate(y_new_x_input_layer[-1]))


        # 将上一个的激活输出当做输入
        y_new_x_input_last = np.dot(new_x_input_layer_act[-1], weight_middle[-1])

        # 这次的输出进行激活
        new_x_input_last_act = activate(y_new_x_input_last)



        new_x_h = new_x_input_last_act

    return y_new_x_input_layer, new_x_h,weight_middle,new_x_input_layer_act,y_new_x_input_last


y_new_x_input_layer, y_pre,weight_middle ,new_x_input_layer_act,y_new_x_input_last= rnn_forward_net()

delta_last = [-0.05,0.02,0.03,-0.06]

print(len(weight_middle))
print(weight_middle)
print()
# print(delta_list[-1])
m=4
def back_prop():
    delta_list = []
    delta_list.append(delta_last)
    count=1

    for i in range(len(x_input)):


        for j in range(len(weight_middle)-1,0,-1):
            print(j)
            # print(delta_list[-1])
            # print(weight_middle[j].T)

            # print(derive_act(y_new_x_input_layer[-count]))
            # print(y_new_x_input_last)
            # print(delta_list[-1])
            # print(derive_act(y_new_x_input_layer[-count]))


            #激活函数的导数该如何带进去呢？？？？？
            # *derive_act(y_new_x_input_layer[-count])
            delta_list.append(np.dot(delta_list[-1], weight_middle[j].T))
            count+=1

        #之前矩阵不是合并了吗，现在把这份delta分开算，delta_left是上一个输出的误差，而上一个输出的误差会继续进行梯度传递
        # print(delta_list[-1])
        delta_left=np.dot(delta_list[-1], weight_middle[0].T)[:m]
        # delta_right=np.dot(delta_list[-1], weight_middle[0].T)[m:]

        delta_list.append(delta_left)


    delta_list.reverse()


    #更新权重
    learning_rate=0.01
    count=1
    for j in range(len(x_input)):
        for i in range(0,len(weight_middle)):

            weight_middle[i] += learning_rate *np.dot(np.array([delta_list[count]]).T,[new_x_input_layer_act[count-1]]).T

            count+=1




back_prop()
