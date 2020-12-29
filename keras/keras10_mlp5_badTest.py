#실습
# r2 : 0.5이하/ 음수 안돼
# layer : 5 이상
# node : 각 10개 이상
# batch_size :8 이하
# epochs : 30이상

# -> validationo_split =0.2로 두자.

import numpy as np
# 1. 데이터
x = np.array([range(100), range(201, 301), range(401,501),
              range(601,701),range(801,901)])
y = np.array([range(811,911),range(1,101)])

# range 값이 같아도 상관없다. 

print(x.shape)  # (5,100)
print(y.shape)  # (2,100)
x_pred2 = np.array([100,302,502,702,1001])
print("x_pred2.shape : ", x_pred2.shape)

x = np.transpose(x) 
y = np.transpose(y)      
# x_pred2 = np.transpose(x_pred2)
x_pred2 = x_pred2.reshape(1, 5)

print(x.shape)    #(100,5)
print(y.shape)    #(100,2)
print(x_pred2.shape)
print("x_pred2.shape : ", x_pred2.shape)  #(1,5)

# (5,)은 1차원, (1,5)는 2차원


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(
    x,y, random_state=66, train_size=0.8, shuffle=True)

print(x_train.shape)   #(80,5)
print(y_train.shape)    #(80,2)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # 텐서플로우에서 케라스 부르는게 속도 더 빠름
# from keras.layers import Dense -> 원래는 이렇게 썼는데 텐서플로우가 케라스 먹음. 이건 좀 느림

model = Sequential()
model.add(Dense(10, input_dim=5))  # 컬럼=피처=특성=열
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(2))

# input과 output은 데이터에 맞춰서 해야함. 히든레이어는 수정가능

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=80, batch_size=8, validation_split=0.2)



# 4. 평가 , 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# input_dim=5 x의 열 값
# Dense의 마지막 output값은 y의 열값
# loss :  1.1731470017650736e-08
# mae :  7.531345181632787e-05
# RMSE :  0.00010831191160387863
# R2 :  0.9999999999851562

y_pred2 = model.predict(x_pred2)
print(y_pred2)
# loss :  8.901626991075773e-09
# mae :  7.717758126091212e-05
# RMSE :  9.434843349127707e-05
# R2 :  0.9999999999887368
# [[992.8982   68.80997]]