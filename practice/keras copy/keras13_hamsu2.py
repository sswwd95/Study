
import numpy as np
# 1. 데이터

x = np.array([range(100), range(201, 301), range(401,501),
              range(601,701),range(801,901)])
y = np.array([range(811,911),range(1,101)])

print(x.shape)  # (5,100)
print(y.shape)  # (2,100)
x_pred2 = np.array([100,302,502,702,1001])
print("x_pred2.shape : ", x_pred2.shape)

x = np.transpose(x) 
y = np.transpose(y)      
# x_pred2 = np.transpose(x_pred2)
x_pred2 = x_pred2.reshape(1, 5)

print(x.shape)    #(100,5)
print(y.shape)    #(100,2)
print(x_pred2.shape)
print("x_pred2.shape : ", x_pred2.shape)  #(1,5)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(
    x,y, random_state=66, train_size=0.8, shuffle=True)

print(x_train.shape)   #(80,5)
print(y_train.shape)    #(80,2)


# 2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Input 

input1 = Input(shape=(5,)) 
aaa = Dense(5, activation='relu')(input1)
aaa = Dense(3)(aaa)
aaa = Dense(4)(aaa)
outputs = Dense(2)(aaa)
model = Model(inputs = input1, outputs = outputs)
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
         validation_split=0.2, verbose=0)


# 4. 평가 , 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

y_pred2 = model.predict(x_pred2)
print(y_pred2)
