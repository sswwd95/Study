import numpy as np

# 1. 데이터
from tensorflow.keras.datasets import cifar10
(x_train,y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000,32,32,3)/255.
x_test = x_test.reshape(10000,32,32,3)/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

ResNet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3))
ResNet50.trainable = False

model = Sequential()
model.add(ResNet50)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience=10, mode='max')
model.fit(x_test, y_test, epochs=100, validation_split=0.2, callbacks=[es],batch_size=16)

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test, batch_size=16)
print('loss, acc : ', loss,acc)
y_pred = model.predict(x_test)

# VGG16
# loss, acc :  1.0346275568008423 0.8866999745368958

# VGG19
# loss, acc :  1.0713889598846436 0.904699981212616

# Xception
# Input size must be at least 71x71; got `input_shape=(32, 32, 3)

# ResNet50
# loss, acc :  1.6861326694488525 0.3921999931335449
