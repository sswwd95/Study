# 2번 파일 코드 복사
# 딥하게 구성 : 모델을 2개 구성(1개는 원칙적 오토인코더, 1개는 마음대로 딥하게 구성)

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train,_),(x_test,_) = mnist.load_data()
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784)/255

# print(x_train[0])
# print(x_test[0])

from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input

# 오토인코더(레이어 구성할 때 노드 (동일하게)오목하게 만들기)
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units= hidden_layer_size,activation='relu', input_shape=(784,)))
    model.add(Dense(800, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(800, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=784) # 중간 레이어를 작게 잡을 수록 전달하는 값이 소멸된다.

'''
# 내 맘대로
model = Sequential()
model.add(Dense(1000, activation='relu', input_shape=(784,)))
model.add(Dense(900, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(784, activation='sigmoid'))
'''

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.compile(optimizer='adam', loss='mse', metrics=['acc'])

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6,ax7, ax8, ax9, ax10)) = \
    plt.subplots(2,5, figsize=(20,7))

# 이미지 5개를 무작위로 고른다. 
random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0:
        ax.set_ylabel('input', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28),cmap='gray')
    if i ==0:
        ax.set_ylabel('output', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# 오토인코더
# 1875/1875 [==============================] - 3s 2ms/step - loss: 0.0803 - acc: 0.0159

# 내 맘대로
# 1875/1875 [==============================] - 4s 2ms/step - loss: 0.0776 - acc: 0.0161

# 둘의 성능차이 별로 없다.