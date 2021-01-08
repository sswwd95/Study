
데이터가 너무 딱 맞으면 새로운 데이터가 들어왔을 때 성능 떨어진다. - > 과적합

과적합이 안되려면?
1. train data 많아야 한다. 
2. 피쳐를 줄인다. (y값이 많다는 건 선이 많아진다는 것-> 과적합 될 가능성이 커진다.)
  - 피쳐가 많을수록 성능이 좋아진다. 줄이라는 건 너~무 많을 경우 줄이라는 것
    ex) 700개에서 100개로 줄여도 성능 똑같음. 나머지 600개는 필요없음
3. regularization(정규화)
4. dropout(딥러닝 해당, 1~3번은 머신러닝도 포함)
   - 노드 중 몇 개를 사용하지 않는 것(삭제하는 것 아님)
(5. 앙상블 : 2~5% 향상된다는 말이 있다. (신뢰없음))
 
##### dropout #####
넣고 싶은 곳에 넣는 것. 
다 넣는 것 아님!

#2 . 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
a = [0.2,0.3,0.4]
model = Sequential()
model.add(Dense(128, activation ='relu', input_shape = (13,)))
model.add(Dropout(a)) # 0.2로 한번 돌리고 0.3으로 한번 돌리고..이런식
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#2 . 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
a = [0.2,0.3,0.4]
b = [0.1,0.2,0.3]
model = Sequential()
model.add(Dense(128, activation ='relu', input_shape = (13,)))
model.add(Dropout(a)) # 0.2로 한번 돌리고 0.3으로 한번 돌리고..이런식
model.add(Dense(128, activation='relu'))
model.add(Dropout(b)) # 이렇게도 가능
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#2 . 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
a = [0.2,0.3,0.4]
b = [0.1,0.2,0.3]
c = [100,200,300]

model = Sequential()
model.add(Dense(128, activation ='relu', input_shape = (13,)))
model.add(Dropout(a)) 
model.add(Dense(c, activation='relu'))# 이렇게도 가능
model.add(Dropout(b)) 
model.add(Dense(c, activation='relu'))
model.add(Dropout(a))
model.add(Dense(c, activation='relu'))
model.add(Dropout(b)) 
model.add(Dense(c, activation='relu'))
model.add(Dense(c, activation='relu'))
model.add(Dense(1))

#2 . 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
a = [0.2,0.3,0.4]
b = [0.1,0.2,0.3]
c = [100,200,300]
d = ['relu','linear','elu','selu','tanh']
model = Sequential()
model.add(Dense(128, activation =d, input_shape = (13,))) # 이렇게도 가능
model.add(Dropout(a)) 
model.add(Dense(c, activation=d))
model.add(Dropout(b)) 
model.add(Dense(c, activation=d))
model.add(Dropout(a))
model.add(Dense(c, activation=d))
model.add(Dropout(b)) 
model.add(Dense(c, activation=d))
model.add(Dense(c, activation=d))
model.add(Dense(1))

# 모델만 경우의 수가 엄청나게 늘어남. 
