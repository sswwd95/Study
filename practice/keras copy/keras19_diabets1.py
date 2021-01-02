#실습 : 19_1,2,3,4,5 Early stopping까지 총 6개의 파일 완성하기

# 1. 데이터
import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:10])
print(x.shape, y.shape) # (442, 10) (442,)

print(np.max(x),np.min(y)) 
print(dataset.feature_names)
print(dataset.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66)

# 2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(356, activation='relu', input_shape=(10, )))
model.add(Dense(128, activation='relu'))


model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, validation_split=0.2, epochs=500, batch_size=8)

#3. 평가, 예측

loss,mae = model.evaluate(x_test, y_test, batch_size=8)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("R2 : " , r2)


# loss, mae :  3918.458251953125 51.98662567138672
# RMSE :  62.59759042244219
# R2 :  0.5076496894168653