# 인공지능계의 hello world라 불리는 mnist
# MNIST 데이터란 필기 숫자의 분류를 위한 학습 데이터 집합. 
# 즉, 이 데이터는 어지럽게 필기된 숫자가 어떤 숫자에 해당하는지 정확하게 맞추기 위한 학습을 위한 것

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) -> 28*28= 28*28*1 같다. 흑백
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])

print