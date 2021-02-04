import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
# stratifiedkfold = k-fold가 label을 데이터와 학습에 올바르게 분배하지 못하는 경우를 해결해준다.  
from keras import Sequential
from keras.layers import *
# math 모듈의 모든 변수, 함수, 클래스 가져온다.
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

train = pd.read_csv('../dacon7/train.csv')
test = pd.read_csv('../dacon7/test.csv')
sub = pd.read_csv('../dacon7/submission.csv')

#distribution of label('digit') 
train['digit'].value_counts()

# drop columns
train2 = train.drop(['id','digit','letter'],1)
test2 = test.drop(['id','letter'],1)

# convert pandas dataframe to numpy array
train2 = train2.values
test2 = test2.values

plt.imshow(train2[100].reshape(28,28))

# reshape
train2 = train2.reshape(-1,28,28,1)
test2 = test2.reshape(-1,28,28,1)

# data normalization
train2 = train2/255.0
test2 = test2/255.0

# imagedatagenerator로 data 늘리기,부풀리기
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1),rotation_range=5)
'''
idg = ImageDataGenerator(rotation_range=회전하는 범위(단위: degree),
                            width_shift_range=수평 이동하는 범위(이미지 가로폭에 대한 비율),
                            height_shift_range=수직 이동하는 범위(이미지의 세로폭에 대한 비율),
                            shear_range=전단(shearing)범위. 크게 하면 더 비스듬하게 찌그러진 이미지가 됨(단위:degree),
                            zoom_range=이미지를 확대/축소시키는 비율(최소:1-zoom_range, 최대:1+zoom_range),
                            channel_shift_range=입력이 RGB3채널인 이미지의 경우 R,G,B각각에 임이의 값을 더하거나 뺄 수 있음(0~255),
                            horizontal_flip=True로 설정 시 가로로 반전,
                            vertical_flip=True로 설정 시 세로로 반전,
                            brightness_range = 이미지의 밝기를 랜덤으로 다르게 준다)

# 문자랑 숫자는 방향성이 있기 때문에 큰 변화 주지 않는다. 
# idg = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
# 표준화는 개별 특징에 대해 평균을 0으로, 분산을 1로 하여 특징별 데이터 분포를 좁히는 방법
# idg = ImageDataGenerator(zca_whitening=True)
# 백색화는 데이터 성분 사이의 상관관계를 없애는 방법.
# 백색화를 수행하면 전체적으로 어두워지고 가장자리가 강조된 것처럼 보이지만 이는 백색화가
# 주위의 픽셀 정보로부터 쉽게 상정되는 색상은 무시하는 효과가 있기 때문. 정보량이 많은 가장자리 등을 강조함으로써 학습 효율을 높일 수 있다.

'''
idg2 = ImageDataGenerator()

# show augmented image data 
sample_data = train2[100].copy()
sample = expand_dims(sample_data,0)
sample_datagen = ImageDataGenerator(rotation_range=5,height_shift_range=(-1,1), width_shift_range=(-1,1))
sample_generator = sample_datagen.flow(sample, batch_size=1)

plt.figure(figsize=(16,10))

for i in range(9) : 
    plt.subplot(3,3,i+1)
    sample_batch = sample_generator.next()
    sample_image=sample_batch[0]
    plt.imshow(sample_image.reshape(28,28))

# Validation

# cross validation
skf = StratifiedKFold(n_splits=40, random_state=42, shuffle=True)

# Modeling
# %%time

reLR = ReduceLROnPlateau(patience=100,verbose=1,factor=0.5) #learning rate scheduler
es = EarlyStopping(patience=160, verbose=1)

val_loss_min = []
result = 0
nth = 0

for train_index, valid_index in skf.split(train2,train['digit']) :
    
    mc = ModelCheckpoint('best_cvision.h5',save_best_only=True, verbose=1)
    
    x_train = train2[train_index]
    x_valid = train2[valid_index]    
    y_train = train['digit'][train_index]
    y_valid = train['digit'][valid_index]
    
    train_generator = idg.flow(x_train,y_train,batch_size=8)
    valid_generator = idg2.flow(x_valid,y_valid)
    test_generator = idg2.flow(test2,shuffle=False)
    
    model = Sequential()
    
    model.add(Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1),padding='same'))
    model.add(BatchNormalization())
    # 배치 정규화 :  미니배치 학습을 통해 배치마다 표준화를 수행하는 것
    # 활성화 함수 relu 등 출력값의 범위가 한정되지 않은 함수의 출력에 배치 정규화를 사용하면 학습이 원활하게 진행되어 큰 효과 발휘
    # 올바르게 정규화하면 활성화 함수에 sigmoid가 아닌 relu함수를 사용해도 좋은 학습 결과 나온다.

    model.add(Dropout(0.2))
    
    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(10,activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None),metrics=['acc'])
    #sparse_categorical_crossentropy : 다중분류 손실 함수. categorical_crossentropy와 동일하지만 원핫인코딩안해도 된다. 
    # epsilon : 0으로 나누어지는 것을 방지

    learning_history = model.fit_generator(train_generator,epochs=2000, validation_data=valid_generator, callbacks=[es,mc,reLR])
    
    # predict
    model.load_weights('best_cvision.h5')
    result += model.predict_generator(test_generator,verbose=True)/40
    
    # save val_loss
    hist = pd.DataFrame(learning_history.history)
    val_loss_min.append(hist['val_loss'].min())
    
    nth += 1
    print(nth, '번째 학습을 완료했습니다.')

print(val_loss_min, np.mean(val_loss_min))

model.summary()

# Submission

sub['digit'] = result.argmax(1)
sub.to_csv('Dacon_cvision_0914_40_epsNone.csv',index=False)




'''
# validation 생성
x_train, x_val, y_train, y_val = train_test_split(
    train2, train['digit'], test_size=0.2, random_state=77, stratify=train['digit'])
# stratify : default=none. target으로 지정해주면 각각의 class비율(ratio)을 train/validation에 유지
# 한 쪽에 쏠려서 분배되는 것을 막아주기 때문에 이 옵션을 사용하지 않으면 classification 문제일 경우 성능차이가 많이 난다.
'''