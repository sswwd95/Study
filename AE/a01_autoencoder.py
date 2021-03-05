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

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img) # 히든레이어가 1개인 모델
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

autoencoder.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 784)]             0
_________________________________________________________________
dense (Dense)                (None, 64)                50240
_________________________________________________________________
dense_1 (Dense)              (None, 784)               50960
=================================================================
Total params: 101,200
Trainable params: 101,200
Non-trainable params: 0
_________________________________________________________________
'''

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train,x_train, epochs=30, batch_size=256, validation_split=0.2) # y는 x와 동일하기 때문에 x_train을 두 번 넣어준다

decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)     # 위에 10개 출력
    plt.imshow(x_test[i].reshape(28,28)) # 원래 x_test의 이미지 출력
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)  # 밑에 10개 출력
    plt.imshow(decoded_imgs[i].reshape(28,28)) # decode된 이미지 출력
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()