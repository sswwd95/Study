# 교차검증 KFOLD (데이터가 독립적이고 동일한 분포를 가질 경우)
'''
train test split의 문제점은?
test 부분의 데이터가 낭비된다. 
-> 데이터 전체를 임의로 몇 등분 나눠서 일부분만 test로 잡고 모든 부분 돌아갈 수 있도록 모델 반복 돌려준다.
그 중 제일 좋은 값 선택
'''

# iris = 다중분류

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score #교차 검증 값
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

print(x.shape,y.shape) 
# 꽃이 3 종류(y값이 3개)
# 0=1 0 0, 1=0 1 0, 2 = 0 0 1

# 머신러닝은 원핫인코딩 필요없음.

kfold = KFold(n_splits=5,shuffle=True) # train과 test 전체를 5조각으로 나눈다

# 2. 모델구성 
model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()  

scores = cross_val_score(model,x, y, cv=kfold) # 모델과 데이터를 엮어준다.
# fit과 scores 다 돌려진 것
print('scores :', scores) #5조각 내서 하나를 test또는 val한 것.
# scores : [0.93333333 0.86666667 0.96666667 1.         0.93333333]

'''
# 3. 훈련
model.fit(x,y)

#4. 평가, 예측

result = model.score(x_test,y_test) 
print('result : ', result)

y_predict = model.predict(x_test) 
# print(y_predict)
print(np.argmax(y_predict,axis=-1))

acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ',acc)

# result = model.score(x_test,y_test) == acc = accuracy_score(y_test, y_predict) 값이 동일함.
'''
