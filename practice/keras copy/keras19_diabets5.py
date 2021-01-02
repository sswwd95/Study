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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.4, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_var = scaler.transform(x_val)

# 2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(10, )))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'min')
model.fit(x_train, y_train, validation_data=(x_val, y_val),
         callbacks=[early_stopping], epochs= 1000, batch_size=1)

#3. 평가, 예측

loss,mae = model.evaluate(x_test, y_test, batch_size=1)
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
# loss, mae :  3294.67236328125 47.2490234375
# RMSE :  57.399236652507014
# R2 :  0.49234948231926967

# 2번 
# loss, mae :  6419.54931640625 68.53453826904297
# RMSE :  80.12209294065812
# R2 :  0.010861353306766519

# 3번
# loss, mae :  3554.2548828125 47.87287521362305
# RMSE :  59.6175707874796
# R2 :  0.45235244649926887

#4번 
# loss, mae :  3308.770263671875 47.48754119873047
# RMSE :  57.521912182867695
# R2 :  0.4901772288329366