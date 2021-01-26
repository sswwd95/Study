'''
##########################################################################################################
numpy란?
- Numpy는 C언어로 구현된 파이썬 라이브러리로써, 고성능의 수치계산을 위해 제작.
- 벡터 및 행렬 연산에 있어서 매우 편리한 기능을 제공
- 데이터분석을 할 때 사용되는 라이브러리인 pandas와 matplotlib의 기반으로 사용되기도 한다.
- 기본적으로 array라는 단위로 데이터를 관리하며 이에 대해 연산을 수행. array는 말그대로 행렬이라는 개념으로 생각
###########################################################################################################
'''
import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
# from tensorflow.keras import models
# from tensorflow import keras
from tensorflow.keras.layers import Dense #가장 기본적인 층

# input_dim 에서 dim은 dimension 차원

model = Sequential() #  -> 모델을 순차적으로 구성
model.add(Dense(5, input_dim=1, activation='linear')) 
model.add(Dense(3, activation='linear'))     
model.add(Dense(4))
model.add(Dense(1)) #-> output 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #최소의 loss를 구하는 건 mse
model.fit(x, y, epochs=1000, batch_size = 1)


#4. 평가, 예측
loss = model.evaluate(x,y, batch_size=1) #  x,y를 넣으면 답안지를 보고 평가하는 것과 같다. 
print("loss : ", loss)

x_pred = np.array([4])
result = model.predict(x_pred)   

# result = model.predict([4])
print('result : ', result)

