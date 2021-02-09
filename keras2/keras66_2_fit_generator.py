import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D,Flatten

train_datagen = ImageDataGenerator(
    rescale=1./255,  # 전처리개념
    horizontal_flip=True,  # True하면 가로로 반전
    vertical_flip=True, # True하면 세로로 반전
    width_shift_range=0.1, # 수평이동
    height_shift_range=0.1, # 수직이동
    rotation_range=5, # 회전
    zoom_range=1.2, # 확대, 축소
    shear_range=0.7, # 크게 하면 더 비스듬하게 찌그러진 이미지가 된다. 
    fill_mode='wrap' # 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식.  
                        # { "constant", "nearest", "reflect"또는 "wrap"} 중 하나
                        # 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                        # 'nearest': aaaaaaaa|abcd|dddddddd
                        # 'reflect': abcddcba|abcd|dcbaabcd
                        # 'wrap': abcdabcd|abcd|abcdabcd
                        # 0으로하면 빈자리를 0으로 채워준다(padding과 같은 개념)? -> 넣어보고 체크하기
)

test_datagen=ImageDataGenerator(rescale=1./255) # 하나를 255로 나눈다는 것

#train_generater
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',
    target_size=(150,150),  
    batch_size=5,
    class_mode='binary'
)

#test_generater
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size=(150,150), 
    batch_size=5,
    class_mode='binary'
)

model = Sequential()
model.add(Conv2D(128,3, padding='same',input_shape=(150,150,3)))
model.add(Conv2D(64,3, padding='same'))
# model.add(Conv2D(64,3,padding='same'))
# model.add(Conv2D(32,3,padding='same'))
model.add(Conv2D(32,3,padding='same'))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1, activation='sigmoid'))

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10,factor=0.5,verbose=1)


model.compile(loss ='binary_crossentropy', optimizer='adam',metrics=['acc'])

# fit은 x, y를 따로 줘야한다. 
# fit_generator은 x,y를 통으로 인식
history = model.fit_generator(
    xy_train, steps_per_epoch=32, epochs=1000, validation_data=xy_test, validation_steps=24, callbacks=[es,lr]
)
# steps per epoch : 전체 데이터의 갯수에서 배치사이즈로 나눈 수를 넣어줘야한다. (전체 train 데이터 수 / 배치사이즈)
# 31로 넣으면?? 160개의 데이터를 배치사이즈 5로 나누면 첫번째 행의 값이 32개가 나오는데 그럼 하나 덜 훈련하는 것.
# validation_steps도 동일.

# history넣어서 반환하여 그림그려보기
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print('acc : ', acc[-1])
print('val_acc : ', acc[:-1])

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss,acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()

# batch size = 5
# fill_mode = nearest
# acc :  0.731249988079071
# val_acc :  [0.5562499761581421, 0.5, 0.4937500059604645, 0.4937500059604645, ''' 0.699999988079071]

# fill_mode = reflect
# acc :  0.737500011920929
# val_acc :  [0.48124998807907104, ''' 0.706250011920929, 0.6875]

# fill_mode = wrap
# acc :  0.699999988079071
# val_acc :  [0.5, 0.5, 0.625, 0.5562499761581421,''' 0.75]