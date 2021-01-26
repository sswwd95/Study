import numpy as np

x= np.load('../data/npy/wine_x.npy')
y= np.load('../data/npy/wine_y.npy')

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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k50_wine_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor = 'acc', patience = 20, mode='max')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', mode='auto', save_best_only=True)
model.fit(x_train, y_train, callbacks=[early_stopping,cp], validation_data=(x_val,y_val), batch_size=8, epochs=1000)

#. 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=8)
print('loss, acc : ', loss,acc)

y_predict = model.predict(x_test[-5:-1])
print(y_predict)
print(np.argmax(y_predict,axis=-1))

# loss, acc :  5.370936924009584e-05 1.0
# [[1.7862796e-08 1.0000000e+00 6.4787256e-09]
#  [4.5854573e-08 1.0000000e+00 1.5934205e-08]
#  [2.4951777e-07 9.9999964e-01 1.0185723e-07]
#  [1.3007568e-09 1.9347868e-09 1.0000000e+00]]
# [1 1 1 2]