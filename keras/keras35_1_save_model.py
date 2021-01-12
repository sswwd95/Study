import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#2. 모델
model = Sequential()
model.add(LSTM(200, input_shape=(4,1)))
model.add(Dense(100))
model.add(Dense(110))
model.add(Dense(110))
model.add(Dense(110))
model.add(Dense(1))

model.summary()

# 데이터 없어도 모델은 돌아간다.
# 모델 save 하는 이유? 재사용하기 위해

# 모델 저장
# model.save("./")  . 은 현재폴더라는 뜻 (Study 폴더)
model.save("../data/h5/save_keras35.h5") #h5는 확장자
model.save("../data/h5//save_keras35_1.h5")
model.save("..\data/h5\save_keras35_2.h5")
model.save("..\\data/h5\\save_keras35_3.h5")
# 위에 4개 다 똑같이 실행됨
# '', "" 상관없음
# \n 이면 줄바꾸기된다. n 같이 쓰려면 \\ 2개 넣어주기
# . 이면 현재 폴더(=작업폴더), ..이면 아래 폴더(c 드라이브)