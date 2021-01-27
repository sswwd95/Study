# 머신러닝은 장비의 제약이 없다. 시간이 빠르다. 
# 딥러닝과 머신러닝을 구별하고, LinearSVC로 문법구조 비교함.

# 머신러닝 : 데이터 -> 모델구성 -> 훈련 -> 평가,예측
# 딥러닝 : 데이터 -> 모델구성 -> 컴파일, 훈련 -> 평가, 예측

# iris = 다중분류

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#classfier = 분류모델
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  # 단순한 데이터를 대상으로 분류할 때 사용 =KNN
from sklearn.linear_model import LogisticRegression # 회귀일 것 같지만 분류모델이다.
from sklearn.tree import DecisionTreeClassifier #scaling의 영향을 받지 않는다.
from sklearn.ensemble import RandomForestClassifier #훈련 과정에서 구성한 다수의 결정 트리들을 랜덤하게 학습시켜 분류 또는 회귀의 결과도출에 사용


#1. 데이터

dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape) # (150,4)
print(y.shape) # (150,)
print(x[:5])
print(y)
# 꽃이 3 종류(y값이 3개)
# 0=1 0 0, 1=0 1 0, 2 = 0 0 1

# 머신러닝은 원핫인코딩 필요없음.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 66)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.4, shuffle=True)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_var = scaler.transform(x_val)

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_var = scaler.transform(x_val)

# 2. 모델구성 -> 통상적으로 밑으로 갈 수록 좋다고 하는 모델.
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier() # ==> rf = RandomForestClassifier(), 'model'은 변수명 정의한 것. 


# 3. 훈련

model.fit(x_train,y_train)

#4. 평가, 예측

#result = model.evaluate(x_test,y_test)
result = model.score(x_test,y_test) # evaluate에서 나오는건 loss,acc였는데 score하면 acc바로 나옴. == model.score는 evaluate의 metrics=acc값과 같다. 
print('result : ', result)

# sklearn 나온 이후에 tensorflow 나왔다. 그래서 model.score보다 model.evaluate가 기능이 더 많다. 

y_predict = model.predict(x_test) # predict는 머신러닝과 동일하게 사용
print(y_predict)
print(np.argmax(y_predict,axis=-1))

acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ',acc)

# result = model.score(x_test,y_test) == acc = accuracy_score(y_test, y_predict) 값이 동일함.

###########################
# Tensorflow
# acc : 0.9666666388511658
###########################

# LinearSVC() 
# result : 0.9333333333333333
# result : 0.9

# SVC
# result :  1.0
# result :  0.9666666666666667
# result :  0.8666666666666667

# KNeighborsClassifier
# result :  0.9666666666666667
# result :  0.9333333333333333
# result :  0.9
# result :  0.8666666666666667

# LogisticRegression
# result :  0.9666666666666667
# result :  0.9
# result :  0.9333333333333333

# DecisionTreeClassifier
# result :  0.9
# result :  0.9333333333333333
# result :  0.9666666666666667

# RandomForestClassifier
# result :  0.9666666666666667
# result :  0.8666666666666667
# result :  0.9
# result :  0.9333333333333333

# minmaxscaler와 standardscaler의 차이가 없는 것 같다. 돌릴 때마다 값 다르게 나옴! 최적의 가중치 나오면 모델체크포인트로 저장하기.




