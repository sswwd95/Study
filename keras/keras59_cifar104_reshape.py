# (n,32,32,3)->(n,32,32,3) reshape레이어 사용
# reshape하려면 flatten해줘야한다. 

import numpy as np
# 1. 데이터
from tensorflow.keras.datasets import cifar10
(x_train,y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

# print(x_train[0])
# print(x_test[0])
# print(x_train[0].shape) #(32, 32, 3)
# print(np.max(x_train)) #255 -> 이미지에서 특성 가장 큰 건 255= 가장 밝음(빛의 3원색)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2],3)/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2],3)/255.
print(x_train.shape)

y_train = x_train
y_test = x_test

print(y_train.shape)
print(y_test.shape)  
# (50000, 32, 32, 3)
# (10000, 32, 32, 3)


# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape) #(50000, 10)

#2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Reshape

model = Sequential()
model.add(Dense(200,activation='relu', input_shape=(32,32,3)))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(3072, activation='relu'))
model.add(Reshape((32,32,3)))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='relu'))

model.summary()
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='acc')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience=20, mode='max')
model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es],batch_size=16)

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test, batch_size=16)
print('loss, acc : ', loss,acc)
y_pred = model.predict(x_test)

# cnn
# loss, acc :  1.3659929037094116 0.521399974822998

# dnn
# loss, acc :  1.5606876611709595 0.45190000534057617
