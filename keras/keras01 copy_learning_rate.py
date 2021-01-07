import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense #가장 기본적인 층

# input_dim 이 input, Dense는 output.
# input_dim 에서 dim은 dimension 차원

model = Sequential()
model.add(Dense(5, input_dim=1, activation='linear')) 
model.add(Dense(3, activation='linear'))     
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, SGD
optimizer = Adam(learning_rate = 0.1)
# optimizer = SGD(learning_rate = 0.1)

model.compile(loss='mse', optimizer=optimizer) #최소의 loss를 구하는 건 mse
model.fit(x, y, epochs=1000, batch_size = 1)


#4. 평가, 예측
loss = model.evaluate(x,y, batch_size=1) #  x,y를 넣으면 답안지를 보고 평가하는 것과 같다. 
print("loss : ", loss)

x_pred = np.array([4])
result = model.predict(x_pred)   

# result = model.predict([4])
print('result : ', result)

