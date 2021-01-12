
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

# 1. 데이터

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])
print(x_train[0].shape) #(28,28)
print(np.max)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test= x_test.reshape(10000,28,28,1)/255. 

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #(60000,10)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(9,2))
model.add(Conv2D(8,2))
model.add(Dropout(0.2))
model.add(Flatten()) 
model.add(Dense(40,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.save('../data/h5/k52_1_model1.h5')

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelCheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
hist = model.fit(x_train,y_train, callbacks=[es,cp], epochs=2, validation_split=0.2, batch_size=16)

model.save('../data/h5/k52_1_model2.h5')
model.save_weights('../data/h5/k52_1_weight.h5')
# 2개의 결과가 동일함.

#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=16)
print('model2_loss : ', result[0])
print('model2_acc : ', result[1])

