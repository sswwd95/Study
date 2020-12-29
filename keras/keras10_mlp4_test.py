#실습
# x는 (100,5) 데이터 임의로 구성
# y는 (100,2) 데이터 임의로 구성
# 모델 완성해라


import numpy as np
# 1. 데이터
x = np.array([range(100), range(201, 301), range(401,501), range(601,701),range(801,901)])
y = np.array([range(811,911),range(1,101)])
print(x.shape) 
print(y.shape)   

x = np.transpose(x) 
y = np.transpose(y)      
print(x)
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(
    x,y, random_state=66, train_size=0.8, shuffle=True)

print(x_train.shape)   #(80,3)
print(y_train.shape)    #(80,3)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # 텐서플로우에서 케라스 부르는게 속도 더 빠름
# from keras.layers import Dense -> 원래는 이렇게 썼는데 텐서플로우가 케라스 먹음. 이건 좀 느림

model = Sequential()
model.add(Dense(10, input_dim=5))  # 컬럼=피처=특성=열
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))

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

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# input_dim=5 x의 열 값
# Dense의 마지막 output값은 y의 열값
# loss :  1.1731470017650736e-08
# mae :  7.531345181632787e-05
# RMSE :  0.00010831191160387863
# R2 :  0.9999999999851562