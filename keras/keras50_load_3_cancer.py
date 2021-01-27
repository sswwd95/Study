import numpy as np

x= np.load('../data/npy/cancer_x.npy')
y= np.load('../data/npy/cancer_y.npy')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.2, shuffle=True)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

from tensorflow.keras.utils import to_categorical 

y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)
print(y)
print(x.shape) #(569, 30)
print(y.shape) # (569, 2) -> reshape됨

#2. 모델
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(30,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
modelpath = '../data/modelcheckpoint/k50_cancer.hdf5'
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', mode='auto', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=1000,callbacks=[es,cp,lr], validation_data=(x_val, y_val), batch_size=8)

# model =load_model('../data/modelcheckpoint/k50_cancer.hdf5')
#4. 평가, 예측
loss, acc = model.evaluate(x_test,y_test, batch_size=8)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test)
# print(y_predict)
# print(y_test)
# print(np.argmax(y_predict,axis=-1))

# loss, acc :  0.16521701216697693 0.9649122953414917
# [[8.9435693e-05 9.9991059e-01]
#  [9.9973267e-01 2.6726536e-04]
#  [1.6807689e-03 9.9831921e-01]
#  [6.8398094e-04 9.9931598e-01]]
# [[0. 1.]
#  [1. 0.]
#  [0. 1.]
#  [0. 1.]]
# [1 0 1 1]