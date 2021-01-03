# 다 : 1 mlp

# 실습 : train과 test 분리해서 소스를 완성하라

import numpy as np
# 1. 데이터
x = np.array([range(100), range(301, 401), range(1,101)])
y = np.array(range(711,811))
print(x.shape)  # (3,100) 
print(y.shape)   # (100, ) 
x = np.transpose(x)      
print(x)
print(x.shape)   # (100, 3) 

#train_test_split은 행을 나누는 것. train_size = 0.7 이면 train (70,3), test (30,3)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(
    x,y, random_state=66, train_size=0.8, shuffle=True)

print(x_train.shape)   #(80,3)
print(y_train.shape)    #(80, )

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

model = Sequential()
model.add(Dense(10, input_dim=3))  # 컬럼=피처=특성=열
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

# 4. 평가 , 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

print("mse : ", mean_squared_error(y_predict, y_test))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# loss :  1.1175870895385742e-08
# mae :  9.765625145519152e-05
# RMSE :  0.0001057159916729051
# mse :  1.1175870895385742e-08
# R2 :  0.9999999999858592

# 모델이 별로인데도 값이 잘 나오는 이유 : 
# w =1 로 평균 1임. 데이터가 딱 맞기 때문에..?