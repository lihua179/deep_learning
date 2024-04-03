# -*- coding:utf-8 -*-
"""
日期：2021年09月08日
目的：
"""
# -*- coding:utf-8 -*-
"""
日期：2021年08月22日
目的：添加激活函数
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
#深层rnn权重原理：这一层的输出（来自上一次，或者说图中左边的）和上一层的输出作为列（列表示对这层的输入一一对应的权重），行就是这层的输出
#到时候会将输入拼接为一个行向量，与矩阵相乘。
def wpt_pre(layer, input_len,output_len):
    weight_middle = []

    weight_middle.append(np.random.random([layer[0], layer[0]+input_len]))
    for i in range(1,len(layer)):
        weight_middle.append(np.random.random([layer[i], layer[i] + layer[i-1]]))

    weight_middle.append(np.random.random([output_len, layer[-1]]))

    return weight_middle

import time

time1=time.time()
layer=[2,2,3]
input_len=6
output_len=1

weight_middle=wpt_pre(layer, input_len,output_len)

# print(weight_middle)

# 先初始化第一次计算时的拼接的上一次输出（假设都为0）
input_zero=[]
for i in range(len(layer)):
    input_zero.append(np.zeros(layer[i]))



output_list=[]

#模拟下第一层的输入值
# input_x=np.random.random(input_len)
input_x=np.random.random([4,input_len])

new_x_input_list=[]
#第一层计算
#输入
new_x_input_list.append(new_x_input_fun(input_zero[0],input_x[0]))
#输出
output_list.append(activate(np.dot(new_x_input_list[-1],weight_middle[0].T)))



#第一次拼接都是和0拼接，所以第一次要单独拿出来计算（类似初始化）
for i in range(1,len(weight_middle)-1):
    # print(i)

    new_x_input_list.append(activate(new_x_input_fun(input_zero[i],output_list[-1])))

    output_list.append(np.dot(new_x_input_list[-1],weight_middle[i].T))

    # print(output_list[-1])

#注意到上面i为的权重次数-1，并且y被单独计算，是因为最后一次计算y不需要拼接上一次的输出y，因此可以直接进行矩阵乘法
y=np.dot(output_list[-1],weight_middle[-1].T)

# print(output_list)

y_list=[]

# print(input_x)
#第二次开始不用初始化了就可以重复这一过程
for j in range(1,len(input_x)):

    #把上次的每次矩阵乘法结果进行储存
    last_output_list = output_list

    # 把这次的每次矩阵乘法结果进行初始化
    output_list = []
    # 把第一次的输入进行拼接
    new_x_input_list=[]
    new_x_input_list.append(activate(new_x_input_fun(last_output_list[0], input_x[j])))
    #将第一次拼接好的输入和第一个矩阵进行相乘得到第一次输出
    output_list.append(np.dot(new_x_input_list[-1], weight_middle[0].T))

    # 这里output_list是每次计算出的最新的输出，同时也是下一层的输入中的一部分
    for i in range(1, len(weight_middle) - 1):
        new_x_input_list.append(activate(new_x_input_fun(last_output_list[i], output_list[-1])))

        output_list.append(np.dot(new_x_input_list[-1], weight_middle[i].T))

    #最终的输出值
    y=np.dot(output_list[-1],weight_middle[-1].T)
    print(y)


    y_list.append(y)

print(weight_middle)

y_error=0.5
delta=[]
delta.append(weight_middle[-1]*y_error)
print()
print('output_delta')
print(delta[-1])
print(layer[-1])
delta.append(np.dot(delta[-1],weight_middle[-2]))
print(layer[-2])
print(delta[-1][0])



letf_weight = delta[-1][0][:layer[-2]]
print('letf_weight')
print(letf_weight)
down_weight = delta[-1][0][layer[-2]:]
print('down_weight')
print(down_weight)
print()
print(weight_middle[-3])
delta.append(np.dot(down_weight, weight_middle[-3]))
print(delta[-1])

#
# # print(weight_middle[-2])
# #
# delta.append(np.dot(down_weight, weight_middle[-2]))
# print(delta[-1])
# for i in range(1,len(layer)):
#     print(i)
#
#     letf_weight = delta[-1][:layer[-i]]
#     print('letf_weight')
#     print(letf_weight)
#     down_weight = delta[-1][layer[-i]:]
#     print('down_weight')
#     print(down_weight)
#     print()
#
#     delta.append(np.dot(down_weight, weight_middle[-i-2]))
#     print('delta[-1]')
#     print(delta[-1])



#

#
# letf_weight=delta[-1][:layer[-1]]
# print('letf_weight')
# print(letf_weight)
# down_weight=delta[-1][layer[-1]:]
# print('down_weight')
# print(down_weight)
#
# delta.append(np.dot(down_weight,weight_middle[-3]))
# print()
# print(delta[-1])
#

# letf_weight=delta[-1][:layer[-2]]
# print('letf_weight')
# print(letf_weight)
# down_weight=delta[-1][layer[-2]:]
# print('down_weight')
# print(down_weight)





# for i in range(len(weight_middle)):
#     delta.append(weight_middle[-1]*y_error)


