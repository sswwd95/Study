from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array
# np.array()
# array()

# 1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = array([11,12,13,14,15])
y_test = array([11,12,13,14,15])

x_pred = array([16,17,18])

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1,activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])   # acc로 하면 정확도가 떨어진다. 3.1과 3은 다르기 때문. 그래서 mse사용. 분류모델. 
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test, batch_size=1)
print("mse, mae : ", results)
y_predict = model.predict(x_test)
# print("y_predict : ", y_predict)

# np.sqrt(results[0])

# 사이킷런? sklearn -> 머신러닝의 라이브러리
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))    
print("RMSE : ", RMSE(y_test, y_predict))                 # mse 수치가 너무 클 때 rmse 사용, rmse는 낮을수록 좋다. 
# print("mse : ", mean_squared_error(y_test, y_predict))

print("mse : ", mean_squared_error(y_predict, y_test))
# 실습 :  RMSE를 0.1 이하로 낮추시오.
# epoch 100 에서 1000으로 바꾸고 validation_split = 0.2에서 0.5로 바꿈

