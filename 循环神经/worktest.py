# -*- coding:utf-8 -*-
"""
日期：2021年08月20日
目的：
"""
import numpy as np

# x_input0=np.random.random([1,3])
# print('x_input0')
# print(x_input0)
# w_x=np.random.random([3,4])
#
# print('w_x')
# print(w_x)
# h0=x_input0.dot(w_x)
#
# # print(h0)
# #第一段激活后输出
# h0=activate(h0)
# print('h0')
# print(h0)
#
# #为下一段想加做准备的w_hh
# w_hh=np.random.random([4,4])
#
# #上一段激活后的h0和专门准备的w_hh矩阵相乘获得新的h，称为hh
# hh=h0.dot(w_hh)
#
# print('hh')
# print(hh)
#
#
# #第二段输入
# x_input1=np.random.random([1,3])
# h1=x_input1.dot(w_x)
# print('h1还未和上一个一起相加')
# print(h1)
#
# h1=hh+h1
# print('h1已和上一个一起相加，并激活')
# h1=activate(h1)
# print(h1)





#实际上可以这么做：将激活后的h0和下一段的x_input一起组成新的输入
# 对上一段计算出的h进行矩阵相乘的w_hh矩阵和对输入进行相乘的矩阵w_x可以共同组成新的矩阵


#
#
# x_input0=np.random.random([1,3])
# print('x_input0')
# print(x_input0)
# w_x=np.random.random([4,3])
#
# print('w_x')
# print(w_x)
# h0=x_input0.dot(w_x.T)
#
# # print(h0)
# #第一段激活后输出
# h0=activate(h0)
# print('h0')
# print(h0)
#
#
#
#
# #为下一段想加做准备的w_hh
# w_hh=np.random.random([4,4])
# # h0_w=h0.dot(w_hh)
# # print(h0_w)
#
# # #第二段输入
# x_input1=np.random.random([1,3])
#
# #将激活后的h0与x_input一起连接
# new_x_input=np.hstack((h0,x_input1))
# print('new_x_input')
# print(new_x_input)
#
# #同理矩阵也是
# new_x_w=np.hstack((w_hh,w_x))
# print('new_x_w')
# print(new_x_w)
#
#
# new_h=new_x_input.dot(new_x_w.T)
# print('new_h')
# print(new_h)


#===========打包环节==============


def activate(x):
    return np.tanh(x)

# def h0_compu(x_input_now,w_x):
#     #输出激活后的上一个h0
#     h0 = activate(np.dot(x_input_now,w_x.T))
#     return h0

def new_x_input_fun(h0,x_input1):
    #将上一个计算激活后的h0和当前新的x_input进行合并
    new_x_input = np.hstack((h0, x_input1))
    return new_x_input

def new_x_w(w_hh,w_x):
    #将上一个计算激活后的h0和当前新的x_input进行合并
    new_x_w=np.hstack((w_hh,w_x))
    return new_x_w

def new_x_h_fun(new_x_input,new_x_w):
    #将合并后的输入和矩阵进行矩阵乘法
    new_h=activate(np.dot(new_x_input,new_x_w.T))
    return new_h


def new_x_h_w_fun(new_x_h,w_hhh,w_hhh_last):
    new_x_h_whhh=np.dot(new_x_h,w_hhh)
    new_x_h_whhh_last=np.dot(new_x_h_whhh,w_hhh_last)

    return new_x_h_whhh_last

#初始化
x_input=np.random.random([10,3])

w_x=np.random.random([1,3])

w_hh=np.random.random([1,1])

w_hhh=np.random.random([1,10])

w_hhh_last=np.random.random([10,1])
#连结后新的权重
new_x_w = new_x_w(w_hh, w_x)



def new_h_(x_input,new_x_w):

    new_x_h = np.zeros(w_hh.shape[0])
    new_h_list = []

    for i in range(len(x_input)):
        # 连结输入（上一个输出和新的输入进行连结）
        new_x_input = new_x_input_fun(new_x_h, x_input[i])

        new_x_h = new_x_h_fun(new_x_input, new_x_w)

        new_x_h=new_x_h_w_fun(new_x_h,w_hhh,w_hhh_last)


        new_h_list.append(new_x_h)

    return new_h_list

# 第一个上一个的输出初始化


new_h_list=new_h_(x_input,new_x_w)

print('new_h_list')
print(new_h_list)

#总结：循环神经网络有点arima内味了（马尔科夫链：每个新的值取决于过去的值）