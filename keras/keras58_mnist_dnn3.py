# 차원 그대로 받아서 출력할때  

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
x_test= x_test.reshape(10000,28,28,1)/255. 
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))

y_train = x_train
y_test = x_test

print(y_train.shape) #(60000, 28, 28, 1)
print(y_test.shape)  # (10000, 28, 28, 1) 

#OneHotEncoding
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape) #(60000,10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(28,28,1)))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='relu'))
#Dense는 차원 다 받아서 Flatten없이 해준다. 마지막 아웃풋은 처음 input shape이랑 맞춰서 해준다.

model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
modelpath = '../data/modelcheckpoint/k57_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=10, mode='min')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
# 5번까지 참는데 개선이 없으면 50프로 줄이겠다. verbose1하면 줄어드는게 보임. 리듀스의 5번이 다 끝나면 es로 간다. 
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, callbacks=[es,cp,reduce_lr], epochs=5, validation_split=0.5, batch_size=16)

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test)
print(y_predict[0])
print(y_predict.shape)


# loss, acc :  0.06896616518497467 0.9800999760627747
'''
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

# loss, acc :  0.13789112865924835 0.9778000116348267
'''

