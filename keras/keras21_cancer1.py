# 유방암 예측 모델
import numpy as np
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape) # (569, 30)
print(y.shape)  # (569,) 
print(x[:5])
print(y)

# 전처리 알아서 해 / minmax, train_test_split

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, activation='relu', input_shape=(30,)))
model.add(Dense(1, activation='sigmoid'))
# 히든레이어 없어도 괜찮다. 

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x,y, epochs=100, validation_split=0.2)

loss = model.evaluate(x,y)
print(loss)
