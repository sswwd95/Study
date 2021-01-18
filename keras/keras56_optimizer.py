# 45번 복사
# learning rate(학습률) : 경사하강법 알고리즘은 기울기에 학습률 또는 step size로 불리는 스칼라를 곱해 다음 지점을 결정
# 학습률이 큰 경우 : 데이터가 무질서하게 이탈하며, 최저점에 수렴하지 못함
# 학습률이 작은 경우 : 학습시간이  매우 오래 걸리며, 최저점에 도달하지 못함


import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2. 모델구성
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

#어느 것이 더 좋다는 건 없다. 알아서 판단하기

########## Adam : lr 값이 작을수록 loss줄어든다####### -> 결과에 큰 차이가 없다.
# optimizer = Adam(lr=0.1)
# loss :  0.026579895988106728 결과물 :  [[11.19065]]
# optimizer = Adam(lr=0.01)
# loss :  9.925211088557262e-06 결과물 :  [[11.00563]]
# optimizer = Adam(lr=0.001)
# loss :  2.964739564959018e-12 결과물 :  [[10.999999]]
# optimizer = Adam(lr=0.0001)   -> 너무 값이 작으면 에폭안에 못들어가서 loss값 더 높아짐.
# loss :  4.6291104808915406e-05 결과물 :  [[10.994312]]

########## Adadelta : lr의 값이 작을수록 loss값 높아진다 ########
# optimizer = Adadelta(lr=0.1)
# loss :  0.0021215833257883787 결과물 :  [[11.076269]]
# optimizer = Adadelta(lr=0.01)
# loss :  8.42503723106347e-05 결과물 :  [[10.980311]]
# optimizer = Adadelta(lr=0.001)
# loss :  10.53264045715332 결과물 :  [[5.1820364]]
# optimizer = Adadelta(lr=0.0001)
# loss :  32.01090621948242 결과물 :  [[0.9640129]]

########## Adamax : lr 값이 작을수록 loss값 줄어든다#######
# optimizer = Adamax(lr=0.1)
#loss :  19.70716667175293 결과물 :  [[7.3953986]]
# optimizer = Adamax(lr=0.01)
# loss :  5.972111910904077e-13 결과물 :  [[10.999999]]
# optimizer = Adamax(lr=0.001)
# loss :  3.174890395030161e-08 결과물 :  [[10.9997015]]
# optimizer = Adamax(lr=0.0001)
# loss :  0.002812793245539069 결과물 :  [[10.935155]]

######## Adagrad : lr의 값이 작을수록 loss값 낮아진다 ######
# optimizer = Adagrad(lr=0.1)
# loss :  444.4090881347656 결과물 :  [[40.024315]]
# optimizer = Adagrad(lr=0.01)
# loss :  4.773441833094694e-05 결과물 :  [[10.995755]]
# optimizer = Adagrad(lr=0.001)
# loss :  7.993978215381503e-05 결과물 :  [[10.98547]]
# optimizer = Adagrad(lr=0.0001)
# loss :  0.005191097501665354 결과물 :  [[10.909922]]

######## RMSprop : lr의 값이 작을수록 loss값 낮아진다 ###### -> 처음 값이 별로 좋지 않음. 확 낮아짐
# optimizer = RMSprop(lr=0.1)
# loss :  11389428.0 결과물 :  [[-6872.1284]]
# optimizer = RMSprop(lr=0.01)
# loss :  4.889301776885986 결과물 :  [[6.305419]]
# optimizer = RMSprop(lr=0.001)
# loss :  1.3367929458618164 결과물 :  [[9.014094]]
# optimizer = RMSprop(lr=0.0001)
# loss :  2.181868694606237e-05 결과물 :  [[10.989986]]

######### SGD : lr의 값이 작을수록 loss값 낮아진다 ###### -> 처음 값이 별로 좋지 않음. 확 낮아짐
# optimizer = SGD(lr=0.1)
# loss :  11389428.0 결과물 :  [[-6872.1284]]
# optimizer = SGD(lr=0.01)
# loss :  nan 결과물 :  [[nan]]
# optimizer = SGD(lr=0.001)
# loss :  4.844811769544322e-07 결과물 :  [[10.999757]]
# optimizer = SGD(lr=0.0001)
# loss :  0.0016498796176165342 결과물 :  [[10.952688]]

######### Nadam
# optimizer = Nadam(lr=0.1)
# loss :  13.756734848022461 결과물 :  [[16.604671]]
# optimizer = Nadam(lr=0.01)
# loss :  0.17057374119758606 결과물 :  [[11.248834]]
# optimizer = Nadam(lr=0.001)
# loss :  4.2206239588352124e-13 결과물 :  [[11.]]
# optimizer = Nadam(lr=0.0001)
# loss :  6.704468887619441e-06 결과물 :  [[10.995356]]




model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x,y, batch_size=1)
y_pred = model.predict([11.])
print('loss : ', loss, '결과물 : ', y_pred)
