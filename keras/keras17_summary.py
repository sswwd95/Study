import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1, activation='linear')) 
model.add(Dense(6, activation='linear'))      
model.add(Dense(4, name = 'aa'))
model.add(Dense(1))

model.summary()

# 실습2 + 과제
# ensemble 1,2,3,4에 대해 summary를 
# 계산하고 이해한 것을 과제로 제출할것
#layer을 만들 때 'name'이란 것에 대해 확인하고 설명할 것
# name을 반드시 써야할 때가 있다. 그때를 말해라.