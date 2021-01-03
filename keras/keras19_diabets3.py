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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

print(np.max(x), np.min(x))
print(np.max(x[0]))


from sklearn.model_selection import train_test_splitS
x_train, x_test, y_train, y_test = train_test_splitS(
    x, y, train_size = 0.8, random_state = 66)

# 2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(10, )))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, validation_split=0.2, epochs=300, batch_size=8)


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

# 1번
# loss, mae :  3768.1259765625 49.35535430908203
# RMSE :  61.385062871563136
# R2 :  0.4193986921311813

# 2번 
# loss, mae :  5818.3115234375 65.9879379272461
# RMSE :  76.27786132701927
# R2 :  0.10350140045743106

# 3번
# loss, mae :  3742.401611328125 49.28392028808594
# RMSE :  61.1751674463525
# R2 :  0.4233624319656353