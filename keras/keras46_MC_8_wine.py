from sklearn.datasets import load_wine
import numpy as np

dataset = load_wine()
print(dataset.DESCR)
# :Number of Instances: 178 (50 in each of three classes)
# :Number of Attributes: 13 numeric, predictive attributes and the class 
# 178행, 13열
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x)
print(y)
print(x.shape) #(178,13)
print(y.shape) #(178,)

# 실습, DNN

from tensorflow.keras.utils import to_categorical

y = to_categorical(y)
print(y.shape) #(178,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size = 0.8, random_state = 55
)

x_train,x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                test_size = 0.2, shuffle = True)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

#2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(300, activation = 'relu', input_shape=(13,)))
model.add(Dense(200, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(3, activation= 'softmax'))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
modelpath = './modelcheckpoint/k46_wine_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor = 'loss', patience=20, mode='min') 
model.fit(x_train, y_train, batch_size = 8, callbacks=[early_stopping, cp], epochs=100, validation_split=0.2)


#. 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=8)
print('loss, acc : ', loss,acc)

y_predict = model.predict(x_test[-5:-1])
print(y_predict)
print(np.argmax(y_predict,axis=-1))


# loss, acc :  0.00012851390056312084 1.0
# [[8.1283273e-09 1.0000000e+00 3.7419234e-10]
#  [2.8588536e-07 9.9999976e-01 2.1940119e-08]
#  [2.6080270e-06 9.9999702e-01 4.0747332e-07]
#  [1.3668525e-09 2.1839366e-09 1.0000000e+00]]
# [1 1 1 2]