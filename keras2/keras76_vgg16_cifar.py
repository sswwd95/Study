import numpy as np
# 1. 데이터
from tensorflow.keras.datasets import cifar10
(x_train,y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

print(x_train[0])
print(x_test[0])
print(x_train[0].shape) #(32, 32, 3)
print(np.max(x_train)) #255 -> 이미지에서 특성 가장 큰 건 255= 가장 밝음(빛의 3원색)

x_train = x_train.reshape(50000,32,32,3)/255.
x_test = x_test.reshape(10000,32,32,3)/255.

print(x_train.shape)

from tensorflow.keras.applications.vgg16 import preprocess_input
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #(50000, 10)

#2. 모델구성
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

VGG16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
VGG16.trainable = False

model = Sequential()
model.add(VGG16)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience=10, mode='max')
model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es],batch_size=16)

# 4. 평가, 예측
loss,acc = model.evaluate(x_test,y_test, batch_size=16)
print('loss, acc : ', loss,acc)
y_pred = model.predict(x_test)

# cnn
# loss, acc :  1.3659929037094116 0.521399974822998

# VGG16.trainable = False
# loss, acc :  1.1224437952041626 0.6090999841690063

# VGG16.trainable = True
# loss, acc :  2.302762031555176 0.10000000149011612

# dense 레이어 노드 수정
# loss, acc :  0.7491508722305298 0.7613000273704529

# dense 레이어 relu 제거
# loss, acc :  1.1773576736450195 0.5831999778747559

# epoch = 100
# loss, acc :  1.0346275568008423 0.8866999745368958


################################### trainable = False ######################################
# VGG16
# loss, acc :  1.7574269771575928 0.35839998722076416

# VGG19

# Xception

# ResNet50

# ResNet101

# InceptionV3

# InceptionResNetV2

# DenseNet121

# MobileNetV2

# NASNetMobile

# EfficientNetB0

################################### trainable = True ######################################
# VGG16

# VGG19

# Xception

# ResNet50

# ResNet101

# InceptionV3

# InceptionResNetV2

# DenseNet121

# MobileNetV2

# NASNetMobile

# EfficientNetB0





