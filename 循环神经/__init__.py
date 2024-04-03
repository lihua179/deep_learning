# -*- coding:utf-8 -*-
"""
日期：2021年08月21日
目的：
"""
import os
# import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from numba import cuda
print(cuda.is_available())
# print(tf.test.is_gpu_available())
# print(torch._)
# import numpy as np
# # import cuda
#
# import numba
# from numba import cuda
# import time as tt
# t1=tt.time()
#
#
# @cuda.jit(nopython=True)
# def add(x):
#     c=0
#
#     for i in range(int(1e10)):
#         c+=x
#     return c
# for j in range(int(1e5)):
#     c = add(10)
#
# print(c)
# print(tt.time()-t1)


# import torch
# info=torch.cuda.is_available()
# import tensorflow as tf
#
# print(info)
# print()
# print(tf.config.list_physical_devices('GPU'))
# print(tf.test.is_gpu_available)
# import numpy as np
#
# from numba import generated_jit, types
# from numba import vectorize
# from numba import jit
#
# from numba import njit, prange
#
# @njit(parallel=True)
# def prange_test(A):
#     s = 0
#     # Without "parallel=True" in the jit-decorator
#     # the prange statement is equivalent to range
#     for i in prange(A.shape[0]):
#         s += A[i]
#     return s
#
# A=np.arange(int(1e6))
# s=prange_test(A)
# print(s)
# print(f1)




