import numpy as np

x= np.load('../data/npy/iris_x.npy')
y= np.load('../data/npy/iris_y.npy')

from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)

print(y)
print(x.shape) #(150,4)
print(y.shape) # (150,3) -> reshape됨

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.4, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_var = scaler.transform(x_val)

# 2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(4,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k50_iris_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', mode='auto', save_best_only=True)
model.fit(x_train,y_train, epochs=500,callbacks=[es,cp], validation_data=(x_val, y_val), batch_size=8)

#4. 평가, 예측

loss,acc = model.evaluate(x_test, y_test, batch_size=8)
print("loss, acc : ", loss, acc)

y_predict = model.predict(x_test[-5:-1])
print(y_predict)

print(y_test[-5:-1])
print(np.argmax(y_predict,axis=-1))

# loss, acc :  0.1411678045988083 0.9666666388511658
# [[3.1532974e-08 4.4133905e-03 9.9558651e-01]
#  [9.9696535e-01 3.0308650e-03 3.8181406e-06]
#  [9.5316672e-01 4.6040565e-02 7.9267478e-04]
#  [2.1009386e-04 3.4959525e-01 6.5019470e-01]]
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# [2 0 0 2]