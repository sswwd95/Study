# dnn

# EarlyStopping

import numpy as np

from sklearn.datasets import load_boston

#1. 데이터
dataset=load_boston()
x = dataset.data
y = dataset.target

print(x.shape) # (506, 13)
print(y.shape)  # (506, )
print("=================")
print(x[:5])  # 0~4까지  -> x 1개당 [6.3200e-03 1.8000e+01 2.3100e+00 0.0000e+00 5.3800e-01 
#                                   6.5750e+00 6.5200e+01 4.0900e+00 1.0000e+00 2.9600e+02 1.5300e+01 3.9690e+02 4.9800e+00] 13개씩 들어있음. 
print(y[:10])
print(np.max(x), np.min(x)) # 711.0  0,0
print(dataset.feature_names)
print(np.max(x[0]))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.7, random_state = 66)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2 . 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, activation ='relu', input_shape = (13,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
modelpath = './modelcheckpoint/k46_boston_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor = 'loss', patience=20, mode='min') # mode에 acc면 'max' /loss면 최솟값이 나와야하니까 min. 헷갈려? 그럼 'auto'넣어

model.fit(x_train, y_train, batch_size = 8, callbacks=[early_stopping, cp], epochs=2000, validation_split=0.2)



# 4. 평가 예측
loss,mae = model.evaluate(x_test, y_test, batch_size=8)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# early stopping
# loss, mae :  11.313929557800293 2.3684475421905518
# RMSE :  3.3636184411913357
# R2 :  0.8630558464213824