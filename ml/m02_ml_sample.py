# 머신러닝은 장비의 제약이 없다. 시간이 빠르다. 
# 딥러닝과 머신러닝을 구별하고, LinearSVC로 문법구조 비교함.

# 머신러닝 : 데이터 -> 모델구성 -> 훈련 -> 평가,예측
# 딥러닝 : 데이터 -> 모델구성 -> 컴파일, 훈련 -> 평가, 예측

import numpy as np
from sklearn.datasets import load_iris

#1. 데이터

# x,y = load_iris(return_X_y=True) 아래와 같은 방법인데 아래 방법이 더 좋다. 
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape) # (150,4)
print(y.shape) # (150,)
print(x[:5])
print(y)
# 꽃이 3 종류(y값이 3개)
# 0=1 0 0, 1=0 1 0, 2 = 0 0 1

# 머신러닝은 원핫인코딩 필요없음.
'''
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
'''
# 2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC 
#svm = support vector machine / 기계 학습의 분야 중 하나로 패턴 인식, 자료 분석을 위한 지도 학습 모델이며, 주로 분류와 회귀 분석을 위해 사용한다.

# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4,)))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(3, activation='softmax'))

model = LinearSVC() # LinearSVC는 강한 모델은 아님. 

# 3. 컴파일
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs= 100, batch_size=8)

model.fit(x,y)

#4. 평가, 예측
# loss,acc = model.evaluate(x_test, y_test, batch_size=8)
# print("loss, acc : ", loss, acc)

#result = model.evaluate(x,y)
result = model.score(x,y) # evalutate에서 나오는건 loss,acc였는데 score하면 acc바로 나옴.
print(result)
#0.9666666666666667


y_predict = model.predict(x[-5:-1]) # predict는 머신러닝과 동일하게 사용
print(y_predict)
print(y[-5:-1])
print(np.argmax(y_predict,axis=-1))
#결과치 나오게 코딩할것.   #argmax

# 딥러닝
# loss, acc :  0.09609115868806839 0.9666666388511658
# [[1.61500391e-09 3.95920433e-05 9.99960423e-01]
#  [9.99981046e-01 1.89530183e-05 1.12919353e-12]
#  [9.84727979e-01 1.52718639e-02 1.03585734e-07]
#  [7.72912681e-05 4.75085050e-01 5.24837613e-01]]
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# [2 0 0 2]

# 머신러닝 기본값으로 했을 때: 0.9666666666666667

