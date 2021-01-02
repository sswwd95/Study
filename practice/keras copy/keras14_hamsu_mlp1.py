# 다 : 1 mlp 함수형
# keras10_mlp2.py를 함수형으로 바꾸시오.

import numpy as np
# 1. 데이터
x = np.array([range(100), range(301, 401), range(1,101)])
y = np.array(range(711,811))
print(x.shape)  
print(y.shape)   

x = np.transpose(x)      
print(x)
print(x.shape)    

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(
    x,y, random_state=66, train_size=0.8, shuffle=True)

x_train,x_val,y_train,y_val = train_test_split(
    x_train,y_train,test_size = 0.2,random_state=66)

print(x_train.shape)  
print(y_train.shape)    

# 2. 모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input

input1=Input(shape=(3,))
dense1 = Dense(3, activation='relu')(input1)
dense2 = Dense(5)(dense1)
dense3 = Dense(5)(dense2)
outputs = Dense(1)(dense3)
model = Model(inputs = input1, outputs = outputs)
model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1, validation_data=(x_val,y_val))


# 4. 평가 , 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

print("mse : ", mean_squared_error(y_predict, y_test))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
