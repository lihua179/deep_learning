# -*- coding:utf-8 -*-
"""
日期：2021年08月22日
目的：
"""
import numpy as np

# #输入
# input_=np.random.random(3)
#
# #对输入x的m层矩阵
# m=6
#
#
# #n=2输出
# n=2
# matrix_x=np.random.random([m,len(input_)])
# matrix_h=np.random.random([m,n])
#
#
# print(input_)
# print(matrix_x)
# print(matrix_h)
# #合并矩阵
# matrix_two=np.hstack([matrix_h,matrix_x])
# print(matrix_two)
#
# output_=np.random.random(n)
# #合并输入
# input_two=np.hstack([output_,input_])
# print(input_two)
#
# #第一层矩阵相乘
# matrix_two_1=np.dot(input_two,matrix_two.T)
# print(matrix_two_1)
#
# #第二层矩阵层数
# m2=3
# matrix_1=np.random.random([len(matrix_two_1),m2])
# print(matrix_1)

# left_output=
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


def wpt_pre(layer, input_len,output_len):
    weight_middle = []

    weight_middle.append(np.random.random([layer[0], layer[0]+input_len]))
    for i in range(1,len(layer)):
        weight_middle.append(np.random.random([layer[i], layer[i] + layer[i-1]]))

    weight_middle.append(np.random.random([output_len, layer[-1]]))

    return weight_middle

import time

time1=time.time()
layer=[2,4,3,4,2]
input_len=3
output_len=5

weight_middle=wpt_pre(layer, input_len,output_len)


input_zero=[]
for i in range(len(layer)):
    input_zero.append(np.zeros(layer[i]))


output_list=[]

input_x=np.random.random(input_len)

new_x_input_list=[]
#第一次计算
new_x_input_list.append(new_x_input_fun(input_zero[0],input_x))
output_list.append(np.dot(new_x_input_list[-1],weight_middle[0].T))
for i in range(1,len(weight_middle)-1):
    # print(i)

    new_x_input_list.append(new_x_input_fun(input_zero[i],output_list[-1]))

    output_list.append(np.dot(new_x_input_list[-1],weight_middle[i].T))

    print(output_list[-1])
y=np.dot(output_list[-1],weight_middle[-1].T)


input_x=np.random.random([2,3])
y_list=[]

for j in range(1,len(input_x)):

    #把上次的每次矩阵乘法结果进行储存
    last_output_list = output_list

    # 把这次的每次矩阵乘法结果进行初始化
    output_list = []
    # 把第一次的输入进行拼接
    new_x_input_list=[]
    new_x_input_list.append(new_x_input_fun(last_output_list[0], input_x[j]))
    #将第一次拼接好的输入和第一个矩阵进行相乘
    output_list.append(np.dot(new_x_input_list[-1], weight_middle[0].T))


    for i in range(1, len(weight_middle) - 1):
        new_x_input_list.append(new_x_input_fun(last_output_list[i], output_list[-1]))

        output_list.append(np.dot(new_x_input_list[-1], weight_middle[i].T))

    #最终的输出值
    y=np.dot(output_list[-1],weight_middle[-1].T)
    print(y)


    y_list.append(y)

print(time.time()-time1)