import numpy as np

x= np.load('../data/npy/diabetes_x.npy')
y= np.load('../data/npy/diabetes_y.npy')

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.4, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_var = scaler.transform(x_val)

# 2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10, )))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
modelpath = '../data/modelcheckpoint/k50_diabetes_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor = 'loss', patience=20, mode='min') 
model.fit(x_train, y_train, batch_size = 8, callbacks=[early_stopping, cp], epochs=200, validation_split=0.2)

#4. 평가, 예측

loss,mae = model.evaluate(x_test, y_test, batch_size=1)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print("R2 : " , r2)

# loss, mae :  3183.80224609375 45.72997283935547
# RMSE :  56.425192802824355
# R2 :  0.5094325786699055