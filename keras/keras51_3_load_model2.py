
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

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape) #(60000,10)

#2. 모델
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# model.summary()

#3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = '../data/modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
# es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
# cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
# hist = model.fit(x_train,y_train, callbacks=[es,cp], epochs=2, validation_split=0.2, batch_size=16)

model = load_model('../data/h5/k51_1_model2.h5')
# 훈련의 결과 = 가중치. 훈련한 다음에 모델 save하면 가중치까지 저장된다. 
# 모델만 저장하고 싶다면 모델 다음에 save하고 컴파일과 훈련 저장하고 싶으면 그 밑에 save
#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print('loss, acc : ', loss, acc)

y_predict = model.predict(x_test[:10])
print(y_predict)
print(y_test[:10])
# loss, acc :  0.06896616518497467 0.9800999760627747
'''
#시각화
import matplotlib.pyplot as plt
plt.rc('font',family='Malgun Gothic') # 한글 폰트 설치

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(hist.history['loss'],marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'],marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('손실비용') 
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right') 

plt.subplot(2,1,2)
plt.plot(hist.history['acc'],marker='.', c='red') 
plt.plot(hist.history['val_acc'],marker='.', c='blue')
plt.grid() 

plt.title('정확도') 
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc','val_acc']) 
plt.show()
'''
