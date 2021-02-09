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

test_datagen=ImageDataGenerator(rescale=1./255)

#train_generater
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',
    target_size=(150,150),  
    batch_size=5,
    class_mode='binary'
    , save_to_dir='../data/image/brain_generator/train/'
)

#test_generater
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size=(150,150), 
    batch_size=5, 
    class_mode='binary'
)