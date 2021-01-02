
from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(1,101))
y = np.array(range(1,101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, train_size=0.8, shuffle=False) 

x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size=0.4, shuffle = False) 

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1, activation = 'relu'))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)  
print(y_predict)


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


