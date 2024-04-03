# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.image
# from matplotlib.pyplot import MultipleLocator
# from PIL import Image
# import math
# import torch

#
#
# img = cv2.imread(r'C:\png\woxuanguang.png')
#
# img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# def rmse(predictions, targets):
#     return np.sqrt(((predictions - targets) ** 2) / predictions ** 2)
#
#
# def im2double(im):
#     min_val = np.min(im.ravel())
#     max_val = np.max(im.ravel())
#     out = (im.astype('float') - min_val) / (max_val - min_val)
#     return out
#
# out = im2double(img1)
#
#
# def correlation_coefficient(T1, T2):
#     numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
#     denominator = T1.std() * T2.std()
#     if denominator == 0:
#         return 0
#     else:
#         result = numerator / denominator
#     return result


# import time
# wait_time2_real = 0
# max_wait_time = 0
#
# time_counter = 0
# time_diff_sum = 0
# time_begin = time.time()
# out_list = []
# imgabs_list = []
#
# out_num_list=np.linspace(0.9,1.1,2000)
# epochs=2000
# for j in range(epochs):
#     print(j)
#
#     out2 = out_num_list[j]*np.abs(out)  # 输入强度
#     # print(out)
#
#     h, w = out2.shape
#     phase = np.pi * 2j * np.random.rand(h, w)
#     s = out2
#     f = np.ones((h, w))
#     t = s * np.exp(phase)
#     i_list = []
#     n_list = []
#     image2 = np.fft.ifft2(np.fft.ifftshift(t))
#     time_one = time.time()
#     for i in range(100):
#         imgangle = np.angle(image2)
#         image = f * np.exp(imgangle * 1j)
#         image = np.fft.fftshift(np.fft.fft2(image))
#         imgabs = np.abs(image) / np.max(np.max(np.abs(image)))  # 输出强度
#         a = correlation_coefficient(out2, imgabs)
#         b = np.sum(out2)
#         c = np.sum(imgabs)
#         i_list.append(i)
#         n_list.append(1 - a)
#         if a >= 0.995:
#             break
#         else:
#
#             imgangle = np.angle(image2)
#             image2 = np.exp(1j * imgangle)
#             image3 = np.fft.fftshift(np.fft.fft2(image2))
#             imgabs = np.abs(image3) / np.max(np.max(np.abs(image3)))
#             imgangle = np.angle(image3)
#             image3 = np.exp(1j * imgangle)
#             image3 = image3 * (out2 + np.random.rand(1, 1) * (out2 - imgabs))
#             image2 = np.fft.ifft2(np.fft.ifftshift(image3))
#
#
#     out_list.append(out2)
#     imgabs_list.append(imgabs)
#
#     if epochs >= 100:
#         power_10 = int(epochs / 10)
#         power_100 = int(epochs / 100)
#
#         time_two = time.time()
#
#         time_diff = time_two - time_one
#         #
#         wait_time = round(i / power_100, 1)
#
#         wait_time2 = round((100 - wait_time) * time_diff * power_100, 2)
#
#         time_counter += 1
#
#         time_diff_sum += wait_time2
#
#         if time_counter >= 167:
#             wait_time2_real = round(time_diff_sum / time_counter, 1)
#
#             time_diff_sum = 0
#             time_counter = 0
#
#             if (max_wait_time <= wait_time2_real):
#                 max_wait_time = round(wait_time2_real, 1)
#
#         if time_counter >= 100:
#             print('\r预计等待时间：', wait_time2_real, '/', max_wait_time, 's', end='       ')
#
#             print("进度：", '|' * (i // power_10), wait_time, '%', end='')
#
#
#
# out_list = np.array(out_list)
# imgabs_list = np.array(imgabs_list)
#
# np.save('out_list.npy',out_list)
# np.save('imgabs_list.npy',imgabs_list)
#
# out_list1=np.load('out_list.npy')
# print(out_list1)
# print(out_list1.shape)


# -*- coding:utf-8 -*-
"""
日期：2021年08月12日
目的：
"""
print('cnn模型框架')
import numpy as np
import time

import cov
import back_propagation as bp

def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))



def softmax(x):
    softmax = np.exp(x) / np.sum(np.exp(x))

    return softmax

def softmax_drev(y_pre=None,y=None):

    return -np.sum((y*np.log(y_pre)))

out_list1=np.load('../卷积cnn/out_list.npy')
print(out_list1)
print(out_list1.shape)
imgabs_list1=np.load('../卷积cnn/imgabs_list.npy')

# out_list1=out_list1.reshape([2000,118680])
# out_list1=out_list1.reshape([2000,118680])
# out_list1=out_list1[:,:1]

print(out_list1.shape)
print(imgabs_list1.shape)



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
        self.input_flatten_=0
        self.right_counter_list = []
        self.error_counter_list= []
        self.deltas_first=0


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
        # print('np.sum(abs(self.y - self.pre_value))')
        # print(np.sum(abs(self.y - self.pre_value)))
        # print(self.pre_value.shape)
        # print(self.y.shape)
        self.error = (self.y - self.pre_value)
        self.deltas = np.array(self.error *(softmax_drev(y_pre=self.pre_value, y=self.y)))
        self.deltas_first=np.sum(self.deltas)

        print(self.deltas_first)
        # print('self.deltas.shape')
        # print(self.deltas.shape)
        # print('self.deltas')
        # print(np.sum(abs(self.deltas)))
        # print()


    def logic_(self):
        self.act_values[-1] = logistic(self.output)
        self.act_out_put = logistic(self.output)
        self.pre_value = self.act_out_put
        self.error = (self.y - self.pre_value)
        self.deltas = self.error *(logistic_deriv(x=self.pre_value))



    def tanh_(self):
        self.act_values[-1] = tanh(self.output)
        self.act_out_put = tanh(self.output)
        self.pre_value = self.act_out_put

        self.error = (self.y - self.pre_value)

        self.deltas = self.error*(tanh_deriv(self.pre_value))



    def nomal_loss(self):

        self.pre_value = self.output
        self.error = (self.y - self.pre_value)
        self.deltas= self.error *(self.activation_deriv(self.pre_value))

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

        # if max(self.pre_value)==max(self.y):
            # right_count+=1
        # print(self.error)
        # print(self.deltas[-1])
        # print(np.sum(np.abs(self.error)))
        self.error_list.append(np.sum(np.abs(self.error)))
        self.delta_list.append(np.sum(np.abs(self.deltas[-1])))


        # if np.abs(self.error[:].tolist()[0])<0.5:
        #     self.learn_rate=self.learn_rate*0.98

    def back_propa(self):


        for l in range(len(self.layer) - 2, 0, -1):

            self.deltas.append(np.dot(self.deltas[-1], self.weights[l].T) * self.activation_deriv(self.act_values[l - 1]))

        self.deltas.reverse()

        # if np.sum(abs(self.deltas[0]))>0.15:
        #     self.deltas[0] = self.deltas[0]*0
        # print(l)
        # print(self.input_flatten.shape)
        # print(self.deltas[0].shape)
        self.weights[0] += self.learn_rate * np.dot(np.array([self.input_flatten]).T, self.deltas[0])
        self.bias[0] += self.learn_rate * self.bias[0] * (self.deltas[0])

        for i in range(1, len(self.weights)):

            self.weights[i] += self.learn_rate * np.dot(self.act_values[i - 1].T, self.deltas[i])
            self.bias[i] += self.learn_rate * self.bias[i] * (self.deltas[i])


        self.input_flatten = np.dot(self.deltas[0], self.weights[0].T) * self.learn_rate
        # print('delta_input_flatten')
        # print(np.sum(abs(self.input_flatten)))
        # print()

        #截断梯度，防止由于神经元自身造成的误差而影响到卷积核，在过大的梯度下直接进行截断
        #等到梯度逐渐稳定时，意味着神经元逐渐形成较为正确的结构，此时所造成的误差较为稳定，
        #基本有卷积核造成，此时可以传递给卷积层


        # self.input_flatten_=np.sum(abs(self.input_flatten))


    def fit_dnn(self,x_train=[],y_train=[],epochs=10,learning_rate=1e-3,filter=[],bias=[],x_test=[],y_test=[],r_epochs=1):


        wait_time2_real = 0
        max_wait_time = 0

        time_counter = 0
        time_diff_sum = 0
        time_begin = time.time()

        for j in range(r_epochs):

            print('epchs:',j)
            for i in range(epochs):
                time_one = time.time()

                pre_num = np.random.randint(0, epochs)
                matrix = x_train[pre_num]

                # zeors_y = np.zeros([118680])
                # zeors_y[y_train[pre_num]] = 1


                # print(self.y)
                self.y = softmax(y_train[pre_num])

                # cnn卷积部分

                cov_demo = cov.nn(matrix)
                # 建立第一层卷积层
                cov_demo.cov_add(step=2, filter=filter[0], padding=0, bias=bias[0])
                cov_demo.act_add(activate_function='relu', compare_num=0)
                cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='max', pooling_padding_num=0)
                # 建立第一层卷积层
                # cov_demo.cov_add(step=1, filter=filter[1], padding=0, bias=bias[1])
                # cov_demo.act_add(activate_function='relu', compare_num=0)
                # cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='max', pooling_padding_num=0)

                cov_demo.flatten()

                # 输出flatten
                # 卷积前向传播梯度接口
                cov_input_flatten = cov_demo.cov_2D.matrix
                # ------------------
                self.input_flatten = cov_input_flatten

                self.forward_propa()

                self.compu_out()

                self.back_propa()

                cut_delta = 50e5

                np_sum_delta=np.sum(abs(self.input_flatten))
                print('np.sum(abs(self.input_flatten))')
                print(np_sum_delta)
                if np_sum_delta < cut_delta:
                    if abs(self.deltas_first)>5e-16:
                        cov_delta = self.input_flatten

                        delta_matrix, filter, bias = bp.back_prop(delta=cov_delta, shape=cov_demo.matrix_shape,
                                                                  cov_demo=cov_demo,
                                                                  learning_rate=learning_rate)

                    # self.input_flatten = self.input_flatten * 0
                    # 卷积反向传播梯度接口




                self.cov_filter = filter
                self.cov_bias = bias

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


            right_counter = 0
            pre_num2 = 100
            for num in range(pre_num2):

                out_put = demo.predict(x_test[num], loss_name='softmax')

                if np.sum(y_test[num]-out_put)<0.2:
                    right_counter += 1

            print()
            print('训练精度：')
            right_counter_=right_counter / pre_num2
            print(right_counter_)

            self.right_counter_list.append(right_counter_)
            self.error_counter_list.append(1-right_counter_)


            print('filter:')
            print(filter)




        self.filter_counter.append(filter)
        self.bias_counter.append(bias)


        print()
        print('实际耗时：')
        print(time.time() - time_begin)


    def predict(self,x_test,loss_name='tanh'):


        matrix=x_test

        # zeors_y=np.zeros([118680])
        # zeors_y[y_test]=1


        self.y = softmax(y_test[num])

        # cnn卷积部分

        cov_demo = cov.nn(matrix)
        # 建立第一层卷积层
        cov_demo.cov_add(step=2, filter=self.cov_filter[0], padding=0, bias=self.cov_bias[0])
        cov_demo.act_add(activate_function='relu', compare_num=0)
        cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='max', pooling_padding_num=0)

        # cov_demo.cov_add(step=1, filter=filter[1], padding=0, bias=bias[1])
        # cov_demo.act_add(activate_function='relu', compare_num=0)
        # cov_demo.pooling_add(x_n=2, y_n=2, pooling_way='max', pooling_padding_num=0)


        cov_demo.flatten()

        # 输出flatten
        # 卷积前向传播梯度接口
        cov_input_flatten = cov_demo.cov_2D.matrix
        # ------------------
        self.input_flatten = cov_input_flatten

        self.forward_propa()

        self.loss=loss_name
        self.loss_fun()

        predict_value=np.argmax(self.pre_value)

        return predict_value

# x_train=np.load('x_train.npy')
# y_train=np.load('y_train.npy')
# y_train=y_train
#
# x_test=np.load('x_test.npy')
# y_test=np.load('y_test.npy')
# y_test=y_test
# print(y_test)

#先单独进行卷积，判断输出到底有几层
# input_flatten = [5, 6, 1, 1, 1]
imgabs_list1=imgabs_list1.reshape([2000,118680])
x_train=out_list1[:1800]
y_train=imgabs_list1[:1800]
x_test=out_list1[1800:]
y_test=imgabs_list1[1800:]
#全连接层layer
layer = [5476,3000,500,118680]

print(y_train.shape)

epochs=1800
r_epochs=5
if epochs>len(y_train):
    print('训练次数大于训练集数！')

# print(y_test[1000])
demo = dnn(layer,activation='tanh',loss='softmax',learn_rate=0.000205,input_shape=[344,345])
#卷积核，偏置初始化
filter = np.random.random([2,50,50])
# filter[0]=[[1.2,1],[1.5,1.2]]
# filter=np.array([[[1,-1],[-1,1]]])
# filter2=np.random.random([2,2])
# filter=[filter1,filter2]
bias = np.random.random([2])
demo.fit_dnn(x_train=x_train,y_train=y_train,epochs=epochs,learning_rate=5e-9,filter=filter,bias=bias,x_test=x_test,y_test=y_test,r_epochs=r_epochs)



print()

right_counter=0
pre_num=500
# print(x_test[1])
for num in range(pre_num):

    out_put = demo.predict(x_test[num], loss_name='softmax')
    # print('预测值')
    # print(out_put)
    # print('真实值')
    # print(y_test[num])

    if np.sum(y_test[num]-out_put)<0.2:
        right_counter+=1

print('测试精度')
print(right_counter/pre_num)
print(demo.cov_filter)
print(demo.cov_bias)



from matplotlib import pyplot as  plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.plot(demo.error_list)
plt.title('error函数')
plt.figure()
plt.title('loss函数')
plt.plot(demo.delta_list)


plt.figure()
plt.title('cnn正确率：')
plt.plot(demo.right_counter_list)
plt.figure()
plt.title('cnn_loss：')
plt.plot(demo.error_counter_list)
plt.show()


