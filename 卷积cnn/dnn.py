# -*- coding:utf-8 -*-
"""
日期：2021年08月12日
目的：
"""
print('cnn模型框架')


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


import numpy as np
import time

class dnn:
    def __init__(self, layer=None, activation='tanh',loss=None,learn_rate=1e-3):

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

        self.pre_value = self.act_out_put
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

    def fit_dnn(self,x_train=[],y_train=[],epochs=10):


        wait_time2_real = 0
        max_wait_time = 0

        time_counter = 0
        time_diff_sum = 0
        time_begin = time.time()



        for i in range(epochs):

            # =x_train[i]

            self.y=y_train[i]

            time_one = time.time()

            #cnn卷积部分



            #输出flatten


            # 卷积前向传播梯度接口
            cov_input_flatten=[]
            self.input_flatten=cov_input_flatten

            self.forward_propa()

            self.compu_out()

            self.back_propa()

            # 卷积反向传播梯度接口
            cov_delta=self.input_flatten




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

        print()
        print('实际耗时：')
        print(time.time() - time_begin)


    def predict(self,x_test):
        self.input_flatten=x_test

        self.forward_propa()


        return self.act_out_put







input_flatten = [5, 6, 1, 1, 1]
layer = [len(input_flatten), 400, 200,10,5,1]
demo = dnn(layer,activation='tanh',loss='tanh',learn_rate=2e-3)

demo.input_flatten = np.array(input_flatten)

x_train=np.random.random([10000,5])
y_train=np.random.random([10000,5])
demo.fit_dnn(x_train,y_train,epochs=10000)

out=demo.predict(np.random.random(5))

print(out)
from matplotlib import pyplot as  plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(demo.error_list)
plt.title('error函数')
plt.figure()
plt.title('loss函数')
plt.plot(demo.delta_list)
plt.show()


