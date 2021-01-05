# 실습 다:1 앙상블을 구현하시오. 


import numpy as np

#1. 데이터
x1 = np.array([range(100), range(301, 401), range(1,101)])
x2 = np.array([range(101,201), range(411,511),range(100,200)])

y = np.array([range(711,811),range(1,101), range(201,301)])

# y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y = np.transpose(y)
# y2 = np.transpose(y2)

# shape 전부다 100행 3열

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y, shuffle = False, train_size=0.8
)
# from sklearn.model_selection import train_test_split
# x1_train, x1_test, y_train, y_test = train_test_split(
#     x1, y, shuffle = False, train_size=0.8)

# from sklearn.model_selection import train_test_split
# x2_train, x2_test, y_train, y_test = train_test_split(
#     x2, y, shuffle = False, train_size=0.8)
# 이렇게 해도 되지만 자원낭비 느낌..?ㅎ


# 2. 모델구성
from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, LSTM, Conv2D
from tensorflow.keras.layers import Dense, Input

# 모델1
input1 = Input(shape=(3,))
dense1 = Dense(10,activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)
# output1 = Dense(3)(dense1)

# 모델2
input2 = Input(shape=(3,))
dense2 = Dense(10,activation='relu')(input2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
# output2 = Dense(3)(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate
# from keras.layers.merge import concatenate, Concatenate 
# from keras.layers import concatenate, Concatenate

# merge = 합치다
merge1 = concatenate([dense1, dense2]) # 제일 끝의 dense 변수명 넣기
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)
# # 모델 분기1
# output1 = Dense(30)(middle1)
# output1 = Dense(7)(output1)
output1 = Dense(3)(middle1) 

# 모델 분기2
# output2 = Dense(15)(middle1)
# output2 = Dense(7)(output2)
# output2 = Dense(7)(output2)
# output2 = Dense(3)(output2)

# 모델 선언
model = Model(inputs=[input1, input2],
              outputs=output1)
        

model.summary()
'''
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mse')
model.fit([x1_train, x2_train], y_train,
          epochs=10, batch_size=1,
          validation_split=0.2, verbose=1)


#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], 
                      y_test, batch_size = 1)
print("model.metrics_names : ", model.metrics_names)
print(loss)

y_predict = model.predict([x1_test, x2_test])

print("==================")
print("y1_predict : \n", y_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))


# R2구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print("R2 : ", r2)
'''