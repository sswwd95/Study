
# 딥러닝으로 해보자

#svm = support vector machine / 기계 학습의 분야 중 하나로 패턴 인식, 자료 분석을 위한 지도 학습 모델이며, 주로 분류와 회귀 분석을 위해 사용한다.
from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터

# XOR 데이터 (입력값이 같지 않으면 1)
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,1,1,0]

# 2. 모델
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid')) # 0과 1이니까 이진분류, 히든레이어없음.

# 3. 컴파일,훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

# 4. 평가, 예측

y_pred = model.predict(x_data)
print(x_data,'의 예측 결과 : ', y_pred)

result = model.evaluate(x_data, y_data)
print('model.score :', result[1])

# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과 :  [[0.50540084]
#  [0.47524184]
#  [0.48430645]
#  [0.45425013]]
# 1/1 [==============================] - 0s 0s/step - loss: 0.6946 - acc: 0.2500
# model.score : 0.25

# 값이 3번 파일과 같이 제대로 안나온다. 어떻게 해결할까?