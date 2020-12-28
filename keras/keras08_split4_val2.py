#실습. validation_date를 만들것. 슬라이싱 하지말고 train_test_split을 사용할 것


from tensorflow.keras.models import Sequential         #import는 가장 위에 명시, import하면 함수로 됨
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
x_train, x_test, y_train, y_test= train_test_split(x,y, train_size=0.8, shuffle=False)  # train_size = 0.6 -> x_train을 60프로 주겠다. x_test는 40 / 0.8이면 test는 81~100까지 나옴

# shuffle=True가 기본값이다. False하면 순서대로 나옴. 순서대로 하면 범위 맞지 않음. 
# 왜 숫자들이 무작위로 나오는가?? 섞은게 좋다. 왜?? 안섞고 순서대로 하면 범위가 맞지 않다. 훈련데이터셋과 테스트데이터셋의 범위가 달라짐. 섞으면 무작위로 전체 범위에서 뽑는거기 때문에 범위가 같아짐. 

print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train, x_val, y_train, y_val= train_test_split(x_train, y_train,
                                                 test_size=0.2, shuffle = False) # 트레인사이즈는 64개  val은 16개  


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
# RMSE :  0.16741932528015258
# mse :  0.028029230477261535
# R2 :  0.9999588132505404

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
# print("mse : ", mean_squared_error(y_test, y_predict))

print("mse : ", mean_squared_error(y_predict, y_test))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


