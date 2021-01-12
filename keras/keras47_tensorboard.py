#tensorboard는 그래프를 넷상에서 볼 수 있게 한 것. 시각화라 성능과는 상관없음.
'''
# cmd -> 
cd \
cd study
cd graph
dir/w
tensorboard --logdir=.
(텐서보드 빼기 로그dir= .은 현재폴더)
위의 순서대로 입력하고 enter누르기

인터넷 켜서 주소창에 
127.0.0.1:6006 ( 127.0.0.1=> 로컬 주소.내 컴퓨터 주소라는 뜻 
                    : 6006=> 로컬호스트에 텐서보드 번호)
'''

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


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
modelpath = '../data/modelcheckpoint/k47_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#epoch:02d 정수형으로 2자리까지 표현, .4f는 소수 4번째자리까지 나온다.
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
# filepath :  최저점마다 파일을 생성하는데 파일 안에 그 지점의 w값이 들어간다. predict, evaluate 할 때 파일에서 땡겨쓰면 좋다. 가장 마지막이 제일 값이 좋은 것.
tb = TensorBoard(log_dir='../data/graph',histogram_freq=0, write_graph=True, write_images=True)
#log_dir='graph' ='./graph'
hist = model.fit(x_train,y_train, callbacks=[es,cp,tb], epochs=10, validation_split=0.2, batch_size=16)

# 기본은 모델이 완벽해야 모델체크포인트에 저장된게 좋은것. 모델이 안좋으면 쓰레기 안에서 그나마 나은걸 뽑은 것.
'''
log_dir : TensorBoard에서 로그 파일을 저장할 디렉토리의 경로
histogram_freq :모델의 계층에 대한 활성화 및 가중치 히스토그램을 계산할 빈도 (에포크 단위).                           
                0으로 설정하면 히스토그램이 계산되지 않는다.
                히스토그램 시각화를 위해 유효성 검사 데이터 (또는 분할)를 지정해야한다.                          
                *histogram= 통계 등 자료의 빈도 분포 특성을 시각화하는 도구  
write_graph : TensorBoard 에서 그래프를 시각화할지 여부. write_graph가 True로 설정되면 로그 파일이 상당히 커질 수 있다.
write_images : TensorBoard 에서 이미지로 시각화하기 위해 모델 가중치를 쓸지 여부.
'''
#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test[:10])
print(y_predict)
print(y_test[:10])
# loss, acc :  0.06896616518497467 0.9800999760627747

#시각화
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic') # 한글 폰트 설치
plt.figure(figsize=(10,6))  # 판 깔아주는 것.
plt.subplot(2,1,1) #(2행 1열 중 첫번째)
plt.plot(hist.history['loss'],marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'],marker='.', c='blue', label='val_loss')
plt.grid()
# subplot은 두 개의 그림을 그린다는 것. plot은 도화지 하나라고 생각.
plt.title('한글') # plt.title('손실비용')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')


plt.subplot(2,1,2) #(2행 2열 중 2번째)
plt.plot(hist.history['acc'],marker='.', c='red', label='acc') #metrics의 이름과 똑같이 넣기
# 그림보면 갱신되는 점은 그대로 두고 뒤에 값 올라간 점은 없어도 된다. 
plt.plot(hist.history['val_acc'],marker='.', c='blue', label='val_acc')
plt.grid() # 격자. 모눈종이 형태. 바탕을 그리드로 하겠다는 것. 

plt.title('accuracy') # plt.title('정확도')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

# loss, acc :  0.05949907749891281 0.9835000038146973

