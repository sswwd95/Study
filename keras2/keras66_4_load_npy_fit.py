import numpy as np

x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
x_test = np.load('../data/image/brain/npy/keras66_test_x.npy')
y_test = np.load('../data/image/brain/npy/keras66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (160, 150, 150, 3) (160,)
# (120, 150, 150, 3) (120,)

# 실습. 모델을 만들기

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size=0.8, random_state=7
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(128,2,padding='same', input_shape=(150,150,3)))
model.add(Dropout(0.2))
model.add(Conv2D(64,2, padding='same'))
model.add(Conv2D(32,2, padding='same'))
model.add(Conv2D(16,2, padding='same'))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1,activation='sigmoid'))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode='min', patience=20)
lr = ReduceLROnPlateau(moniort='val_loss',factor=0.5,patience=10)

model.compile(loss ='binary_crossentropy', optimizer='adam',metrics=['acc'])
model.fit(x_train, y_train, epochs=1000, callbacks=[es,lr], validation_data=(x_val, y_val) )