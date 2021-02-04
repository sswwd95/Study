# pandas, numpy
import numpy as np
import pandas as pd
# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
# 전처리
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold
# stratifiedkfold = k-fold가 label을 데이터와 학습에 올바르게 분배하지 못하는 경우를 해결해준다.  
from keras.optimizers import Adam
# 모델링
from keras import Sequential
from keras.layers import *
# math 모듈의 모든 변수, 함수, 클래스 가져온다.

from keras.callbacks import EarlyStopping, ModelCheckpoint

train = pd.read_csv('../dacon7/train.csv')
test = pd.read_csv('../dacon7/test.csv')
sub = pd.read_csv('../dacon7/submission.csv')

print(train.shape, test.shape, sub.shape)  
# (2048, 787) (20480, 786) (20480, 2)
# print(train.head())

# 불필요한 컬럼 제거
train2 = train.drop(['id', 'digit', 'letter'],1)
test2 = test.drop(['id', 'letter'],1)

# 데이터 프레임(pd)을 ndarray(np) 형식으로 변환 후 reshape
train2 = train2.values.reshape(-1,28,28,1)
test2 = test2.values.reshape(-1,28,28,1)

print(np.max(train2), np.max(test2)) # 255 255

# data 정규화
train2 = train2/255.0
test2 = test2/255.0

# validation 생성
x_train, x_val, y_train, y_val = train_test_split(
    train2, train['digit'], test_size=0.2, random_state=77, stratify=train['digit'])
# stratify : default=none. target으로 지정해주면 각각의 class비율(ratio)을 train/validation에 유지
# 한 쪽에 쏠려서 분배되는 것을 막아주기 때문에 이 옵션을 사용하지 않으면 classification 문제일 경우 성능차이가 많이 난다.
'''
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(train2[i])
plt.suptitle('orginal image', fontsize=12)
plt.show()
'''

# imagedatagenerator로 data 늘리기,부풀리기
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
'''
# idg = ImageDataGenerator(rotation_range=45, height_shift_range=(-1,1), width_shift_range=(-1,1))
idg = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True,
                        zca_whitening=True, brightness_range=[0.1,1.0],rotation_range=20,height_shift_range=(-1,1), width_shift_range=(-1,1))
# 문자랑 숫자는 방향성이 있기 때문에 큰 변화 주지 않는다. 

# idg = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
# 표준화는 개별 특징에 대해 평균을 0으로, 분산을 1로 하여 특징별 데이터 분포를 좁히는 방법
# idg = ImageDataGenerator(zca_whitening=True)
# 백색화는 데이터 성분 사이의 상관관계를 없애는 방법.
# 백색화를 수행하면 전체적으로 어두워지고 가장자리가 강조된 것처럼 보이지만 이는 백색화가
# 주위의 픽셀 정보로부터 쉽게 상정되는 색상은 무시하는 효과가 있기 때문. 정보량이 많은 가장자리 등을 강조함으로써 학습 효율을 높일 수 있다.
idg2 = ImageDataGenerator()


'''
# 이미지 시각화
sample_data = train2[28].reshape(1,28,28,1)
sample_gen = idg.flow(sample_data, batch_size=1)

plt.figure(figsize=(16,10))

for i in range(9) : 
    plt.subplot(3,3,i+1)
    plt.imshow(sample_gen[0].reshape(28,28))
plt.suptitle('result image', fontsize=12)
plt.show()
'''
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
reLR = ReduceLROnPlateau(patience=15, verbose=1, factor=0.5)
es = EarlyStopping(patience=30, verbose=1)

skf = StratifiedKFold(n_splits=40, random_state=77, shuffle=True)

result = 0
nth=0

# cross validation
for train_index,val_index in skf.split(train2,train['digit']):
    cp = ModelCheckpoint('../dacon7/check/my.h5', save_best_only=True, verbose=1)

    x_train = train2[train_index]
    x_val = train2[val_index]
    y_train = train['digit'][train_index]
    y_val = train['digit'][val_index]

    train_gen = idg.flow(x_train,y_train,batch_size=32)
    val_gen = idg2.flow(x_val, y_val)
    test_gen = idg2.flow(test2,shuffle=False)

    model = Sequential()
    model.add(Conv2D(64,2, activation='relu', input_shape=(28,28,1)))
    model.add(BatchNormalization())
    # 배치 정규화 :  미니배치 학습을 통해 배치마다 표준화를 수행하는 것
    # 활성화 함수 relu 등 출력값의 범위가 한정되지 않은 함수의 출력에 배치 정규화를 사용하면 학습이 원활하게 진행되어 큰 효과 발휘
    # 올바르게 정규화하면 활성화 함수에 sigmoid가 아닌 relu함수를 사용해도 좋은 학습 결과 나온다.
    model.add(Conv2D(64,2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))
    model.add(Conv2D(32,2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,2,padding='same',activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(16,3,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(16,3,padding='same',activation='relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(10,activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001,epsilon=None), metrics='acc')
    #sparse_categorical_crossentropy : 다중분류 손실 함수. categorical_crossentropy와 동일하지만 원핫인코딩안해도 된다. 
    # epsilon : 0으로 나누어지는 것을 방지
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val))
    # 52/52 [==============================] - 1s 17ms/step - loss: 1.8263 - acc: 0.4451 - val_loss: 3.7120 - val_acc: 0.0976
    history.history.keys()

    learning_history = model.fit(train_gen, validation_data=val_gen, epochs=1000, callbacks=[es,cp,reLR])

    # predict
    model.load_weights('../dacon7/check/my.h5')
    result += model.predict_generator(test_gen,verbose=True)/40

    # 학습결과 확인 
    hist = pd.DataFrame(learning_history.history)
    print(hist['val_loss'].min())
    
    nth +=1
    print(nth, '번째 학습을 완료했습니다.')

sub['digit'] = result.argmax(1)
sub.to_csv('../dacon7/sub/my3.csv',index=False)

'''
plt.title('Training and validation loss')
plt.xlabel('epochs')

plt.plot(hist['val_loss'])
plt.plot(hist['loss'])
plt.legend(['val_loss','loss'])

plt.figure()

plt.plot(hist['acc'])
plt.plot(hist['val_acc'])
plt.legend(['acc','val_acc'])
plt.title('Training and validation accuracy')

plt.show()
'''

# 점수 0.1225490196	 개똥망