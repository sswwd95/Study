
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(1,101))
y = np.array(range(1,101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,
                             #train_size=0.9, test_size=0.2, shuffle=True)  
                            train_size = 0.7, test_size=0.2,shuffle=True)
                             # 위 두가지의 경우에 대해 확인 후 정리할 것!
                             # 1.1 이 넘을 경우와 1이 안될 경우 비교 
                             # 1.1 이 넘으면 => The sum of test_size and train_size = 1.1, should be in the (0, 1) range. Reduce test_size and/or train_size.
                             # 1.1 이 안넘으면 => [53 37 16 41 77 65 54 27 45 24 57 93 75 13 71 97 76  9 20 40 38 78  2 66
                                                # 64 15 58 69 21 86 29 81  1 63 39 10 59 62 89 36 51 72 14  3 44 52 26 82
                                                # 6 80 35 95 87 19 17 25 11 92 60 98 67 61 74 18 28 99  8 47 96 83]
                                                # (70,)
                                                # (20,)
                                                # (70,)
                                                # (20,)
print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
'''
x_train, x_val, y_train, y_val= train_test_split(x_train, y_train,
                                                 train_size=0.8, shuffle = True) 


print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)


# (60,) -> 스칼라가 60개. 이건 1차원. dim = 1

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(1))

#model.add(Dense(1))
# model.add(Dense(1, input_dim=1)) -> 이것도 가능. 히든레이어 없어도 실행은 가능함. 대신 성능이 떨어짐.  
# 히든레이어가 없으면 한번만 연산하고 끝남. 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100) # train_size = 0.8 에서 validation 0.2 니까 80개 중에서 20프로만 쓰니까  16

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

#validation = 0.2   -> 성능이 좋아짐, 안좋아질수도 있다. 
# loss :  0.07724357396364212
# mae :  0.22822844982147217

#validation_data 
# loss :  0.022586554288864136
# mae :  0.1324489414691925

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))

print("mse : ", mean_squared_error(y_predict, y_test))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
'''

