
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

# (5,)은 1차원, (1,5)는 2차원


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(
    x,y, random_state=66, train_size=0.8, shuffle=True)

print(x_train.shape)   #(80,5)
print(y_train.shape)    #(80,2)


# 2. 모델구성
from tensorflow.keras.models import Model # 함수 모델 
from tensorflow.keras.layers import Dense, Input 

input1 = Input(shape=(5,))  # Input layer를 직접구성하겠다는것
# inputs 쓰거나 input1 써도 상관없다. 변수명이라서
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1) # 위의 레이어에서 받은 input 변수명 끝에 적기
dense3 = Dense(4)(dense2)
outputs = Dense(2)(dense3)
model = Model(inputs = input1, outputs = outputs)
model.summary()
#히든레이어 변수 바꿔도 결과에서는 dense로 자동으로 잡아서 나타내줌.
'''
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 5)]               0
_________________________________________________________________
dense (Dense)                (None, 5)                 30
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 10
=================================================================
'''

#위의 경우가 함수모델. 밑에 시퀀셜이 시퀀셜 모델. 성능차이는 없다. 표현방식만 다를뿐

# model = Sequential()
# # model.add(Dense(10, input_dim=1))  # 컬럼=피처=특성=열
# model.add(Dense(5, activation='relu', input_shape=(1,)))  
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(2))
# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=5000, batch_size=1,
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
