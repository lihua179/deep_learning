# -*- coding:utf-8 -*-
"""
日期：2021年08月12日
目的：
"""
print('cnn模型框架')

import cov
import back_propagation as bp
import numpy as np
import time

def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))

    # def activation_deriv(self,x=[]):
    #     return 1 / (1 + np.exp(-x))


def softmax(x):
    softmax = np.exp(x) / np.sum(np.exp(x))

    return softmax

def softmax_drev(y_pre=None,y=None):

    return -(y*np.log(y_pre))



class dnn:
    def __init__(self, layer=None, activation='tanh',loss=None,learn_rate=1e-3,input_shape=[5,5]):

        self.layer = layer
        self.weights = []
        self.act_values = []
        self.input_flatten = None
        self.bias = []
        self.pre_value = []
        self.y = []
        self.deltas = []
        self.delta_ = []
        self.error = []
        self.loss=loss
        self.activation = []
        self.values = []
        self.learn_rate=learn_rate
        self.delta_list=[]
        self.error_list=[]
        self.cov_filter=[]
        self.cov_bias=[]
        self.input_shape=input_shape
        self.filter_counter=[]
        self.bias_counter=[]


        self.output = []
        self.input_ = []
        self.weights_ = []
        self.bias_ = []
        self.act_out_put = []

        if activation == 'logic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv

        if activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv


        for i in range(0, len(layer) - 1):
            self.weights.append(np.random.random([layer[i], layer[i + 1]]))
            self.act_values.append(np.zeros([layer[i + 1], 1]))
            self.deltas.append(np.zeros([layer[i + 1], 1]))
            self.bias.append(np.random.random([1, layer[i + 1]]))

    def softmax_(self):
        self.act_values[-1] = softmax(self.output)
        self.act_out_put = softmax(self.output)
        self.pre_value=self.act_out_put
        self.error = (self.y - self.pre_value)
        self.deltas = np.array(self.error *softmax_drev(y_pre=self.pre_value, y=self.y))

    def logic_(self):
        self.act_values[-1] = logistic(self.output)
        self.act_out_put = logistic(self.output)
        self.pre_value = self.act_out_put
        self.error = (self.y - self.pre_value)
        self.deltas = self.error *logistic_deriv(x=self.pre_value)



    def tanh_(self):
        self.act_values[-1] = tanh(self.output)
        self.act_out_put = tanh(self.output)
        self.pre_value = self.act_out_put
        self.error = (self.y - self.pre_value)
        self.deltas = self.error*tanh_deriv(self.pre_value)

    def nomal_loss(self):

        self.pre_value = self.output
        self.error = (self.y - self.pre_value)
        self.deltas= self.error * self.activation_deriv(self.pre_value)

    def loss_fun(self):

        if self.loss is None:
            self.nomal_loss()

        elif self.loss=='softmax':
            self.softmax_()

        elif self.loss=='logic':
            self.logic_()

        elif self.loss=='tanh':
            self.tanh_()

        self.deltas=[self.deltas]

    def forward_propa(self):
        self.act_out_put = self.input_flatten

        for i in range(0, len(self.weights)):
            # 为这次矩阵乘法赋值权重和偏置

            self.output = np.dot(self.act_out_put, self.weights[i]) + self.bias[i]

            self.act_out_put = self.activation(self.output)

            # 记录此次矩阵乘法激活值
            self.act_values[i] = self.act_out_put

    def compu_out(self):

        self.loss_fun()

        self.error_list.append(np.abs(self.error[:].tolist()[0]))
        self.delta_list.append(np.abs(self.deltas[-1].tolist()[0]))


        if np.abs(self.error[:].tolist()[0])<0.5:
            self.learn_rate=self.learn_rate*0.98

    def back_propa(self):


        for l in range(len(self.layer) - 2, 0, -1):

            self.deltas.append(np.dot(self.deltas[-1], self.weights[l].T) * self.activation_deriv(self.act_values[l - 1]))

        self.deltas.reverse()



        self.weights[0] += self.learn_rate * np.dot(np.array([self.input_flatten]).T, self.deltas[0])
        self.bias[0] += self.learn_rate * self.bias[0] * (self.deltas[0])

        for i in range(1, len(self.weights)):

            self.weights[i] += self.learn_rate * np.dot(self.act_values[i - 1].T, self.deltas[i])
            self.bias[i] += self.learn_rate * self.bias[i] * (self.deltas[i])


        self.input_flatten = np.dot(self.deltas[0], self.weights[0].T) * self.learn_rate

    def fit_dnn(self,x_train=[],y_train=[],epochs=10,learning_rate=1e-3):


        wait_time2_real = 0
        max_wait_time = 0

        time_counter = 0
        time_diff_sum = 0
        time_begin = time.time()

        #卷积核，偏置初始化
        filter = np.random.random([4, 4, 4])
        bias = np.random.random([4])

        filter_counter=[]
        bias_counter=[]
        for i in range(epochs):
            time_one = time.time()


            matrix =x_train[i]

            # matrix=matrix.reshape(self.input_shape)

            self.y=y_train[i]

            #cnn卷积部分

            cov_demo = cov.nn(matrix)
            # 建立第一层卷积层
            cov_demo.cov_add(step=1, filter=filter[0], padding=0, bias=bias[0])
            cov_demo.act_add(activate_function='relu', compare_num=0)
            cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='mean', pooling_padding_num=2)
            # # 建立第二层卷积层
            cov_demo.cov_add(step=1, filter=filter[1], padding=2, bias=bias[1])
            cov_demo.act_add(activate_function='relu', compare_num=0)
            cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='max', pooling_padding_num=2)
            # # 建立第三层卷积层
            cov_demo.cov_add(step=1, filter=filter[2], padding=2, bias=bias[2])
            cov_demo.act_add(activate_function='relu', compare_num=0)
            cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='max', pooling_padding_num=2)
            cov_demo.flatten()

            #输出flatten
            # 卷积前向传播梯度接口
            cov_input_flatten=cov_demo.cov_2D.matrix
            #------------------
            self.input_flatten=cov_input_flatten

            self.forward_propa()

            self.compu_out()

            self.back_propa()

            # 卷积反向传播梯度接口
            cov_delta=self.input_flatten

            if np.abs(self.error[:].tolist()[0]) < 0.5:
                learning_rate = learning_rate * 0.98

            delta_matrix, filter, bias = bp.back_prop(delta=cov_delta, shape=cov_demo.matrix_shape, cov_demo=cov_demo,
                                                      learning_rate=learning_rate)

            if epochs >= 100:
                power_10 = int(epochs / 10)
                power_100 = int(epochs / 100)

                time_two = time.time()

                time_diff = time_two - time_one
                #
                wait_time = round(i / power_100, 1)

                wait_time2 = round((100 - wait_time) * time_diff * power_100, 2)

                time_counter += 1

                time_diff_sum += wait_time2

                if time_counter >= 167:
                    wait_time2_real = round(time_diff_sum / time_counter, 1)

                    time_diff_sum = 0
                    time_counter = 0

                    if (max_wait_time <= wait_time2_real):
                        max_wait_time = round(wait_time2_real, 1)

                if time_counter >= 100:
                    print('\r预计等待时间：', wait_time2_real, '/', max_wait_time, 's', end='       ')

                    print("进度：", '|' * (i // power_10), wait_time, '%', end='')

        self.filter_counter.append(filter)
        self.bias_counter.append(bias)
        self.cov_filter=filter
        self.cov_bias=bias

        print()
        print('实际耗时：')
        print(time.time() - time_begin)


    def predict(self,x_test,loss_name='tanh'):


        matrix=x_test
        # matrix = matrix.reshape(self.input_shape)

        self.y = y_test

        # cnn卷积部分

        cov_demo = cov.nn(matrix)
        # 建立第一层卷积层
        cov_demo.cov_add(step=1, filter=self.cov_filter[0], padding=0, bias=self.cov_bias[0])
        cov_demo.act_add(activate_function='relu', compare_num=0)
        cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='mean', pooling_padding_num=2)

        # 建立第二层卷积层
        cov_demo.cov_add(step=1, filter=self.cov_filter[1], padding=2, bias=self.cov_bias[1])
        cov_demo.act_add(activate_function='relu', compare_num=0)
        cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='max', pooling_padding_num=2)
        # # 建立第三层卷积层
        cov_demo.cov_add(step=1, filter=self.cov_filter[2], padding=2, bias=self.cov_bias[2])
        cov_demo.act_add(activate_function='relu', compare_num=0)
        cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='max', pooling_padding_num=2)
        cov_demo.flatten()

        # 输出flatten
        # 卷积前向传播梯度接口
        cov_input_flatten = cov_demo.cov_2D.matrix
        # ------------------
        self.input_flatten = cov_input_flatten

        self.forward_propa()

        self.loss=loss_name
        self.loss_fun()

        return self.act_out_put




x_train=np.load('x_train.npy')
y_train=np.load('y_train.npy')
y_train=y_train/10.0

x_test=np.load('x_test.npy')
y_test=np.load('y_test.npy')
y_test=y_test/10.0
# print(y_test)

#先单独进行卷积，判断输出到底有几层
# input_flatten = [5, 6, 1, 1, 1]

#全连接层layer
layer = [16,32,10,1]


epochs=5000
if epochs>len(y_train):
    print('训练次数大于训练集数！')


demo = dnn(layer,activation='tanh',loss=None,learn_rate=0.55,input_shape=[28,28])
demo.fit_dnn(x_train=x_train,y_train=y_train,epochs=epochs,learning_rate=0.4855)



right_counter=0
pre_num=10
for num in range(pre_num):

    out_put = demo.predict(x_test[num], loss_name=None)
    print('预测值')
    print(out_put)
    print('真实值')
    print(y_test[num])
    # print('错误值')
    # print(y_test[num]-out_put)

    if abs(y_test[num]-out_put)<0.05:
        right_counter+=1

print('正确率')
print(right_counter/pre_num)

print()
print('demo.bias_counter')
print(demo.bias_counter)
print()
print('demo.filter_counter')
print(demo.filter_counter)
print()
print('demo.weights[-1]')
print(demo.weights[-1])

from matplotlib import pyplot as  plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(demo.error_list)
plt.title('error函数')
plt.figure()
plt.title('loss函数')
plt.plot(demo.delta_list)
plt.show()


