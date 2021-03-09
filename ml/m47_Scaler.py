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
# x = x / 711  # 최댓값으로만 나눈다. x의 값이 0~711이면 float형인데  711.하면 실수형으로 된다. 정수면 .안찍어도 상관없지만 형변환
# x = (x - 최소) / (최대 - 최소)
#   = (x - np.min(x)) / (np.max(x) - np.min(x)) 식 자체가 이렇게 되어있다는 것을 이해하기
print(np.max(x[0]))
from sklearn.preprocessing import MaxAbsScaler, PowerTransformer
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,QuantileTransformer
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler() # 이상치 제거
# scaler = QuantileTransformer() 
# scaler = QuantileTransformer(output_distribution='normal') # 정규분포
# scaler = MaxAbsScaler()
# scaler = PowerTransformer(method='yeo-johnson')
scaler = PowerTransformer(method='box-cox') # 데이터가 0이나 마이너스면 못 쓴다
scaler.fit(x)     # -> 이렇게 하면 전체에 전처리 들어가서 예측 값이 범위 벗어나면 값 엉망. x train만 전처리.
x = scaler.transform(x)

# standard
print(np.max(x), np.min(x)) # 711.0  0,0 => 9.933930601860268 -3.9071933049810337
print(np.max(x[0]))  #0.44105193260704206



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
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
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

#전처리 전
# loss, mae :  17.961238861083984 3.0568878650665283
# RMSE :  4.238070188962735
# R2 :  0.7825965976870914

# 전처리 후 x = x/711.
# loss, mae :  13.763806343078613 2.824580430984497
# RMSE :  3.7099605406018363
# R2 :  0.8334024435018605

# MinMaxScaler
# loss, mae :  13.234770774841309 2.4213950634002686
# RMSE :  3.637962471311643
# R2 :  0.8398059151970972

# standard
# loss, mae :  9.964120864868164 2.233564615249634
# RMSE :  3.156599500727736
# R2 :  0.8793939723975026

# robustscaler(중위값을 중심으로 한 이상치 제거에 효과적인 방법이다)
# loss, mae :  9.202962875366211 2.3226966857910156
# RMSE :  3.0336385960390246
# R2 :  0.8886070440741408

# QuantileTransformer(1000개의 분위수(데이터가 1000개 이하면 애매), 평균값이 아니다.robustscaler의 업그레이드 버전)
# loss, mae :  13.331345558166504 2.3808205127716064
# RMSE :  3.651211521519729
# R2 :  0.8386369728037223

# robustscaler, quantiletransformer은 outlier 처리에 효과적
# 하지만 이거 두개만 사용하지는 말 것, outlier해보고 쓰기

# MASABSSCALER
# loss, mae :  11.228340148925781 2.220862865447998
# RMSE :  3.350871502690251
# R2 :  0.8640918206805114

# scaler = PowerTransformer(method='yeo-johnson')
# loss, mae :  12.750399589538574 2.4392049312591553
# RMSE :  3.5707698717013248
# R2 :  0.8456687869028725

# scaler = PowerTransformer(method='box-cox')
# ValueError: The Box-Cox transformation can only be applied to strictly positive data

