from tensorflow.keras.models import Sequential        
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(1,101))
y = np.array(range(1,101))

#주석은 ''' 아니면 """

'''  
x_train = x[:60]  
x_val = x[60 : 80]  
x_test = x[80:]   

 
y_train = y[:60] 
y_val = y[60 : 80]  
y_test = y[80:]  
''' 
# 텐서플로우 안에서 sklearn쓰면 더 강력해짐

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True)  #  x_test는 20/ 0.8이면 81~100까지 나옴


# shuffle=True가 기본값. False하면 순서대로 나옴. 순서대로 하면 범위 맞지 않음. 

print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(x_test.shape)

# (60,) -> 스칼라가 60개. 1차원

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)  # y_predict는 y_test와 근사한 값이 나와야함
print(y_predict)

# 결과치는 주석해서 달아놓기

# shuffle = false
#loss :  0.017558246850967407
#mae :  0.13092155754566193

# Shuffle = true    -> ?  보통 트루가 값이 더 좋게 나옴. 
# loss :  0.17716486752033234
# mae :  0.3572970926761627
