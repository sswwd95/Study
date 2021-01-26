
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
from tensorflow.keras.models import Sequential,load_model
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# model = Sequential()
# model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(9,2))
# model.add(Conv2D(8,2))
# model.add(Dropout(0.2))
# model.add(Flatten()) 
# model.add(Dense(40,activation='relu'))
# model.add(Dense(10,activation='softmax'))

# model.save('../data/h5/k52_1_model1.h5')

# 3. 컴파일, 훈련
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = '../data/modelCheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
# k52_1_mnist_??? => k52_1_MCK.h5 이름을 바꿔줄 것
# es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
# cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
# hist = model.fit(x_train,y_train, callbacks=[es,cp], epochs=2, validation_split=0.2, batch_size=16)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# model.save('../data/h5/k52_1_model2.h5')
# model.save_weights('../data/h5/k52_1_weight.h5')

# model = load_model('../data/h5/k52_1_model.h5')

#4-1. 평가, 예측
# result = model.evaluate(x_test, y_test, batch_size=16)
# print('model1_loss : ', result[0])
# print('model1_acc : ', result[1])

# model.load_weights('../data/h5/k52_1_weight.h5')

#4-2. 평가, 예측
# result = model.evaluate(x_test, y_test, batch_size=16)
# print('가중치_loss : ', result[0])
# print('가중치_acc : ', result[1])
# 가중치_loss :  0.0892680361866951
# 가중치_acc :  0.9729999899864197

model=load_model('../data/modelcheckpoint/k52_1_mnist_checkpoint.hdf5') 
# model=load_model('../data/modelcheckpoint/k52_1_mnist_checkpoint.h5') 

result = model.evaluate(x_test, y_test, batch_size=16)
print('로드체크포인트_loss : ', result[0])
print('로드체크포인트_acc : ', result[1])
# 로드체크포인트_loss :  0.07565636932849884
# 로드체크포인트_acc :  0.9764999747276306

# modelcheckpoint를 load하면 최소의 loss 값 찍은 부분을 불러오는거라 값의 변동없음. 
# 