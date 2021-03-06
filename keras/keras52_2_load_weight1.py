
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

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# fit부분에서 가중치만 저장해서 compile은 명시해야함. 
model.load_weights('../data/h5/k52_1_weight.h5')
#4-2. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=16)
print('가중치_loss : ', result[0])
print('가중치_acc : ', result[1])
# 가중치_loss :  0.0892680361866951
# 가중치_acc :  0.9729999899864197

# 모델~컴파일, 훈련까지 다 저장
model2=load_model('../data/h5/k52_1_model2.h5')
result2 = model2.evaluate(x_test, y_test, batch_size=16)
print('로드모델_loss : ', result2[0])
print('로드모델_acc : ', result2[1])
# 로드모델_loss :  0.0892680361866951
# 로드모델_acc :  0.9729999899864197

# 두 개다 weight값 저장했기 때문에 값 같음