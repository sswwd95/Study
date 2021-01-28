# cancer = 이진분류

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split,KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  # 단순한 데이터를 대상으로 분류나 회귀를 할 때 사용
from sklearn.linear_model import LogisticRegression # 회귀일 것 같지만 분류모델이다.
from sklearn.tree import DecisionTreeClassifier # scaling의 영향을 받지 않는다.
from sklearn.ensemble import RandomForestClassifier #훈련 과정에서 구성한 다수의 결정 트리들을 랜덤하게 학습시켜 분류 또는 회귀의 결과도출에 사용


datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape) # (569, 30)
print(y.shape)  # (569,) 

x_train,x_test,y_train,y_test = train_test_split(
    x, y, random_state=77, shuffle=True, train_size=0.8
)

kfold = KFold(n_splits=5,shuffle=True) # train만 5조각으로 나눈다

# 2. 모델구성

models = [LinearSVC, SVC, KNeighborsClassifier,DecisionTreeClassifier, RandomForestClassifier]
for algorithm in models :  
    model = algorithm()
    scores = cross_val_score(model,x_train, y_train, cv=kfold) 
    print('scores :', scores) 

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
# model = LinearSVC()   
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()  => 값이 제일 좋게 나왔다.

# scores : [0.91208791 0.96703297 0.9010989  0.86813187 0.9010989 ]
# scores : [0.9010989  0.9010989  0.93406593 0.91208791 0.9010989 ]
# scores : [0.94505495 0.91208791 0.92307692 0.95604396 0.89010989]
# scores : [0.92307692 0.96703297 0.89010989 0.94505495 0.93406593]
# scores : [0.95604396 0.94505495 0.95604396 0.95604396 0.98901099]