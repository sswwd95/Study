#체크포인트에 날짜와 시간 표시

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])
print(x_train[0].shape) #(28,28)
print(np.max)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
# x_train의 최댓값 : 255
# .astype('float32') -> 정수형을 실수형으로 바꾸는 것
x_test= x_test.reshape(10000,28,28,1)/255. 
# 이렇게 해도 실수형으로 바로 된다. 
#x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)) -> 코딩할 때 이렇게 쓰기!

#OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #(60000,10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(9,2))
model.add(Conv2D(8,2))
model.add(Dropout(0.2))
model.add(Flatten()) 
model.add(Dense(40,activation='relu'))
model.add(Dense(10,activation='softmax'))


model.summary()
'''
########################################################################################################################################

import datetime # 컴퓨터에서 제공되는 시간과 동일. 클라우드에서 쓰면 미국시간으로 된다. 한국시간으로 바꿔서 잡아주기. 코랩은 영국시간 기준
# 덮어쓰기 할 경우 구분하기 좋다. ..?
date_now = datetime.datetime.now() # 문제점  : 여기 시간으로 고정된다. 분이 넘어가도 수정안됨.
                                   # 체크포인트 내로 now()를 넣어서 수정
print(date_now)
date_time = date_now.strftime('%m%d_%H%M') #strttime = startime # 월, 일, 시간, 분
print(date_time)


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
filepath = '../data/modelcheckpoint/' # 경로 변수만들기
filename = '_{epoch:02d}-{val_loss:.4f}.hdf5' # 파일 이름 변수 만들기
# stirng끼리 합치기 ("". join은 빈 공백 안에 넣는다는 것)
modelpath = "".join([filepath,"k45_",date_time, filename])
print(modelpath) #../data/modelcheckpoint/k45_0127_1018_{epoch:02d}-{val_loss:.4f}.hdf5

# modelpath = '../data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'(기존의 시간 포함 안한 경로)

#########################################################################################################################################
'''
########################################################################################################################################

import datetime # 컴퓨터에서 제공되는 시간과 동일. 클라우드에서 쓰면 미국시간으로 된다. 한국시간으로 바꿔서 잡아주기. 코랩은 영국시간 기준
# 덮어쓰기 할 경우 구분하기 좋다. ..?

                            

date_time = datetime.datetime.now().strftime('%m_%d_%H_%M_%S') #strttime = startime # 월, 일, 시간, 분
print(date_time)


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

filepath = '../data/modelcheckpoint/' # 경로 변수만들기
filename = '_{epoch:02d}-{val_loss:.4f}.hdf5' # 파일 이름 변수 만들기
# stirng끼리 합치기 ("". join은 빈 공백 안에 넣는다는 것)
modelpath = "".join([filepath,"k45_",date_time,filename])
print(modelpath) #../data/modelcheckpoint/k45_0127_1018_{epoch:02d}-{val_loss:.4f}.hdf5

# modelpath = '../data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'(기존의 시간 포함 안한 경로)

#########################################################################################################################################
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
cp = ModelCheckpoint(filepath=modelpath , monitor='val_loss', save_best_only=True, mode='auto')

#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, callbacks=[es,cp], epochs=1000, validation_split=0.2, batch_size=16)


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test[:10])
print(y_predict)
print(y_test[:10])
# loss, acc :  0.06896616518497467 0.9800999760627747

#시각화
import matplotlib.pyplot as plt
plt.rc('font',family='Malgun Gothic') # 한글 폰트 설치

plt.figure(figsize=(10,6))  # 판 깔아주는 것.
plt.subplot(2,1,1) #(2행 1열 중 첫번째)
plt.plot(hist.history['loss'],marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'],marker='.', c='blue', label='val_loss')
plt.grid()
# subplot은 두 개의 그림을 그린다는 것. plot은 도화지 하나라고 생각.
plt.title('손실비용') 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') # 상단의 우측 부분에 라벨 명시를 해주는 것
# legend는 표시해주는 거라 그래프 보고 알아서 위치 설정하기.

plt.subplot(2,1,2) #(2행 2열 중 2번째)
plt.plot(hist.history['acc'],marker='.', c='red') #metrics의 이름과 똑같이 넣기
# 그림보면 갱신되는 점은 그대로 두고 뒤에 값 올라간 점은 없어도 된다. 
plt.plot(hist.history['val_acc'],marker='.', c='blue')
plt.grid() # 격자. 모눈종이 형태. 바탕을 그리드로 하겠다는 것. 

plt.title('정확도') 
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc','val_acc']) # 레전드에 직접 라벨명 넣어줄 수 있다. 위치 알아서 설정함

plt.show()

# loss, acc :  0.05949907749891281 0.9835000038146973

