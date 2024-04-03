# -*- coding:utf-8 -*-
"""
日期：2021年08月23日
目的：
"""
# import os
import tensorflow as tf
# import torch
#
# import torch
# print(torch.cuda.is_available())

# info=torch.cuda.is_available()
# import tensorflow as tf
#
# print(info)
# print()
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available())
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from numba import cuda
# # # # import random
# # # import numpy as np
# # # # import cudatoolkit as cuda
# import time as t
# t1=t.time()
# # # import math
#
# @cuda.jit
# def logistic_regression():
#     print('hi')
#
#     # return Y
#
# logistic_regression()
#
# print(t.time()-t1)




# import torch
# a=torch.cuda.is_available()
# print(a)
# print(torch.__version__)

