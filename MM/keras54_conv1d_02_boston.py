
import numpy as np

from sklearn.datasets import load_boston

#1. 데이터
dataset=load_boston()
x = dataset.data
y = dataset.target

print(x.shape) # (506, 13)
print(y.shape)  # (506, )
print("=================")
print(x[:5]) 
print(y[:10])
print(np.max(x), np.min(x)) # 711.0  0,0
print(dataset.feature_names)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

print(y_train.shape) # (404, 1)

#2 . 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv1D(500,3,input_shape=(x_train.shape[1],1)))
model.add(Flatten())
model.add(Dense(400, activation='relu'))
# model.add(Dropout(0.2)) 
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))



# 3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=20, mode='min') 

model.fit(x_train, y_train, batch_size = 32, callbacks=[es], epochs=2000, validation_split=0.2)


# 4. 평가 예측
loss,mae = model.evaluate(x_test, y_test, batch_size=32)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_test)

# RMSE, R2 = 회귀모델 지표
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# lstm
# loss, mae :  10.443477630615234 2.2721614837646484
# RMSE :  3.2316371347232526
# R2 :  0.8750525058832845

#cnn
# loss, mae :  8.52490520477295 2.2797205448150635
# RMSE :  2.919743971656153
# R2 :  0.8980066370941056

# conv1d
# loss, mae :  5.849340915679932 1.9010809659957886
# RMSE :  2.4185410459383747
# R2 :  0.9300175253751992