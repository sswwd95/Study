#keras67_1 남자 여자에 노이즈 넣어서 복구하기

# 실습. fit 사용

import tensorflow
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization, Activation, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt

x_train = np.load('../data/image/gender/npy/keras67_train_x.npy')
y_train = np.load('../data/image/gender/npy/keras67_train_y.npy')
x_val = np.load('../data/image/gender/npy/keras67_val_x.npy')
y_val = np.load('../data/image/gender/npy/keras67_val_y.npy')


print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
# (1389, 128, 128, 3) (1389,)
# (347, 128, 128, 3) (347,)
print(x_train[0].shape)
# plt.imshow(x_train[0])
# plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size=0.8, random_state=7
)

print(x_train.shape, y_train.shape)
# (1111, 128, 128, 3) (1111,)
print(x_test.shape)
# (278, 128, 128, 3)

# ###################################################################
# x_train = x_train.reshape(1111,128*128*3)
# from sklearn.decomposition import PCA
# pca = PCA(n_components=0.99)
# X_reduced = pca.fit_transform(x_train)  # PCA 계산 후 투영
# print('선택한 차원(픽셀) 수 :', pca.n_components_) 
# 95% : 281, 99% : 692
# ####################################################################
# 몇% 상관없고 내가 알아서 판단


x_train_noised = x_train + np.random.normal(0,0.2, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0,0.2, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) 
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

def autoencoer(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters= hidden_layer_size, kernel_size=3,padding='same', input_shape=(128,128,3), activation='relu'))
    model.add(Conv2D(128,3,padding='same', activation='relu'))
    model.add(Conv2D(64,3,padding='same', activation='relu'))
    model.add(Conv2D(3,3, padding='same',activation='sigmoid'))
    return model

model = autoencoer(hidden_layer_size=281)
model.summary()


model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['acc']
)

from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='loss')
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='loss')
filepath = ('../data/modelcheckpoint/cae-{val_acc:.4f}.hdf5')
cp = ModelCheckpoint(filepath=filepath, save_best_only=True, verbose=1)

history= model.fit(x_train_noised, x_train, epochs=500, callbacks=[es,rl], validation_split=0.2 )
print('loss : ', history.history['loss'][-1])
print('acc : ', history.history['acc'][-1])

model.save('../data/h5/67CAE.h5')

model = load_model('../data/h5/67CAE.h5')

output = model.predict(x_test_noised) # x_test에 대한 노이즈가 제거되었는지 확인


import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6,ax7, ax8, ax9, ax10),  (ax11,ax12, ax13, ax14, ax15)) = \
    plt.subplots(3,5, figsize=(20,7))

# 이미지 5개를 무작위로 고른다. 
random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(128,128,3),cmap='gray')
    if i ==0:
        ax.set_ylabel('input', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(128,128,3),cmap='gray')
    if i ==0:
        ax.set_ylabel('noise', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(128,128,3),cmap='gray')
    if i ==0:
        ax.set_ylabel('output', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


# loss :  0.00012983998749405146
# acc :  1.0
# 1/1 [==============================] - 0s 998us/step - loss: 0.9666 - acc: 0.8571
# loss, acc :  0.966610848903656 0.8571428656578064

# 노이즈 제거
# loss :  0.48343831300735474
# acc :  0.8301087617874146
