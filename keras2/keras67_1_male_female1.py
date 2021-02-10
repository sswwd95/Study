# 실습. fit_generator 사용

import tensorflow
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    BatchNormalization, Activation, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# male = 841
# female = 895

datagen=ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
    zoom_range=0.2,
    rescale=1./255,
    validation_split=0.2  
)

batch_size = 32
train_gen=datagen.flow_from_directory(
    'c:/data/image/gender/',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=batch_size,
    subset="training"
)

val_gen=datagen.flow_from_directory(
    'c:/data/image/gender/',
    target_size=(128, 128),
    class_mode='binary',
    batch_size=batch_size,
    subset="validation"
)



print(train_gen[0][0].shape, val_gen[0][0].shape)
# (1389, 128, 128, 3) (347, 128, 128, 3)


np.save('../data/image/gender/npy/keras67_train_x.npy', arr=train_gen[0][0])
np.save('../data/image/gender/npy/keras67_train_y.npy', arr=train_gen[0][1])
np.save('../data/image/gender/npy/keras67_val_x.npy', arr=val_gen[0][0])
np.save('../data/image/gender/npy/keras67_val_y.npy', arr=val_gen[0][1])

model=Sequential()
model.add(Conv2D(128, 2, padding='same', input_shape=(128, 128, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, 2, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(2, padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['acc']
)

from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
es=EarlyStopping(patience=20, verbose=1, monitor='loss')
rl=ReduceLROnPlateau(patience=10, verbose=1, monitor='loss')
filepath = ('../data/modelcheckpoint/fitgen_gender-{val_acc:.4f}.hdf5')
cp = ModelCheckpoint(filepath=filepath, save_best_only=True, verbose=1)

history = model.fit_generator(
    train_gen,
    steps_per_epoch=train_gen.samples//batch_size,
    epochs=1000,
    callbacks=[es, rl],
    validation_data=val_gen
)

print('loss : ', history.history['loss'][-1])
print('acc : ', history.history['acc'][-1])

# loss :  0.07028840482234955
# acc :  0.9786293506622314

model.save('../data/h5/fitgen.h5')


loss, acc = model.evaluate_generator(val_gen)
print('loss, acc : ', loss, acc)
# loss, acc :  0.5986067652702332 0.7838616967201233

pred = model.predict_generator(val_gen,verbose=True)
print('pred : ',pred)