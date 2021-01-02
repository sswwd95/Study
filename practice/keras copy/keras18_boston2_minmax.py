import numpy as np

from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape) # (506, 13)
print(y.shape)  # (506, )
print("=================")
print(x[:5])  # 0~4까지  -> x 1개당 [6.3200e-03 1.8000e+01 2.3100e+00 0.0000e+00 5.3800e-01 
#                                   6.5750e+00 6.5200e+01 4.0900e+00 1.0000e+00 2.9600e+02 1.5300e+01 3.9690e+02 4.9800e+00] 13개씩 들어있음. 
print(y[:10])

print(np.max(x), np.min(x)) # 711.0  0,0
print(dataset.feature_names)
# print(dataset.DESCR)

# 데이터 전처리(MinMax)
x = x / 711  # 최댓값으로만 나눈다. x의 값이 0~711이면 float형인데  711.하면 실수형으로 된다. 정수면 .안찍어도 상관없지만 형변환
# x = (x - 최소) / (최대 - 최소)
#   = (x - np.min(x)) / (np.max(x) - np.min(x)) 식 자체가 이렇게 되어있다는 것을 이해하기


#전처리 전
# loss, mae :  27.019460678100586 4.129007816314697
# RMSE :  5.198024668725807
# R2 :  0.6729555986669336

# 전처리 후
# loss, mae :  13.402970314025879 2.729325771331787
# RMSE :  3.661006663445475
# R2 :  0.8377700310479839


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.7, random_state = 66
)

#2 . 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(Dense(128, activation = 'relu', imput_dim = 13)) 가능
model.add(Dense(128, activation ='relu', input_shape = (13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, batch_size = 8, epochs=100, validation_split=0.2)



# 4. 평가 예측
loss,mae = model.evaluate(x_test, y_test, batch_size=8)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
