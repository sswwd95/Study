# 실습 : train과 test 분리해서 소스를 완성하라

import numpy as np
# 1. 데이터
x = np.array([range(100), range(301, 401), range(1,101)])
y = np.array(range(711,811))
print(x.shape)  # (3,100) 
print(y.shape)   # (100, ) -> 행렬 아니라 그냥 데이터 값

# x에서 첫번째 컬럼에 있는건 0~99까지 300~400, 1~100까지 들어감.  
x = np.transpose(x)      
print(x)
print(x.shape)    # (100, 3) -> 백터가 1개 스칼라 100개

#train_test_split은 행을 정리하는 것 트레인테스트가 70프로면 100행중에서 70행 3열이 트레인되는 것
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(
    x,y, random_state=66, train_size=0.8, shuffle=True)
#위의 순서대로 해야함.


print(x_train.shape)   #(80,3)
print(y_train.shape)    #(80, )


# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # 텐서플로우에서 케라스 부르는게 속도 더 빠름
# from keras.layers import Dense -> 원래는 이렇게 썼는데 텐서플로우가 케라스 먹음. 이건 좀 느림

model = Sequential()
model.add(Dense(10, input_dim=3))  # 컬럼=피처=특성=열
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#훈련을 쓸거면 훈련할 값을 넣고 fit x,y값이 아니라 x_train

# 4. 평가 , 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)
# 예측을 x를 하는 것이 아니라 x_test값을 넣어야한다.
# x_test를 해보니 어떤 값이 나오니 예측한 것, y_test와 쌍으로 연결

# print(y_predict)

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