
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#epoch:02d 정수형으로 2자리까지 표현, .4f는 소수 4번째자리까지 나온다.
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
# filepath :  최저점마다 파일을 생성하는데 파일 안에 그 지점의 w값이 들어간다. predict, evaluate 할 때 파일에서 땡겨쓰면 좋다. 가장 마지막이 제일 값이 좋은 것.
hist = model.fit(x_train,y_train, callbacks=[es,cp], epochs=2, validation_split=0.2, batch_size=16)

# 기본은 모델이 완벽해야 모델체크포인트에 저장된게 좋은것. 모델이 안좋으면 쓰레기 안에서 그나마 나은걸 뽑은 것.

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test[:10])
print(y_predict)
print(y_test[:10])
# loss, acc :  0.06896616518497467 0.9800999760627747

#시각화
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
path = 'C:\\Users\\bit\Downloads\\nanumbarungothic.ttf'
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

