# keras23_LSTM3_scale을 DNN으로 코딩
# 결과치 비교
# DNN으로 23번 파일보다 loss를 좋게 만들것

# LSTM 코딩. 80을 원한다

import numpy as np
# 1. 데이터
x = np.array([[1,2,3] ,[2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70]) 

print("x.shape : ", x.shape) #(13, 3)
print("y.shape : ", y.shape) #(13,)


# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size = 0.8, random_state = 66)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_dim=3))
model.add(Dense(30, activation='relu'))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience=20, mode='min') 

model.fit(x, y, batch_size = 8, callbacks=[early_stopping], epochs=200, validation_split=0.2)

# 4. 평가, 예측

loss,mae = model.evaluate(x, y, batch_size=8)
print("loss : ", loss)

x_predict = x_predict.reshape(1,3)

result = model.predict(x_predict)
print(result)

# loss :  0.03733702003955841
# [[80.27483]]

