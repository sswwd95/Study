from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

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
model.compile(loss='mse', optimizer='adam', metrics=['mae'])  
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test, batch_size=1)
print("mse, mae : ", results) # compile에 넣어준 loss, metrics값이 나온다. 
# mse, mae :  [0.921644389629364, 0.9432497024536133]
y_predict = model.predict(x_test)
# print("y_predict : ", y_predict)

# np.sqrt(results[0])

# 사이킷런? sklearn -> 머신러닝의 라이브러리
from sklearn.metrics import mean_squared_error
# RMSE = 평균 제곱근 오차, Root mean squared error /회귀 분석을 쓸 때 가장 많이 쓰는 평가 지표. MSE에 루트를 씌운 것. 낮을수록 정밀도가 높다. MSE값이 너무 클 때 사용한다. 
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))   # np.sqrt = SQuare RooT의 약자 = 제곱근
print("RMSE : ", RMSE(y_test, y_predict))                 
# RMSE :  0.9600233570599057
print("mse : ", mean_squared_error(y_predict, y_test))
# mse :  0.9216448461005712


