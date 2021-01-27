# wine = 다중분류

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  # 단순한 데이터를 대상으로 분류나 회귀를 할 때 사용
from sklearn.linear_model import LogisticRegression # 회귀일 것 같지만 분류모델이다.
from sklearn.tree import DecisionTreeClassifier # scaling의 영향을 받지 않는다.
from sklearn.ensemble import RandomForestClassifier #훈련 과정에서 구성한 다수의 결정 트리들을 랜덤하게 학습시켜 분류 또는 회귀의 결과도출에 사용


dataset = load_wine()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x)
print(y)
print(x.shape) #(178,13)
print(y.shape) #(178,)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size = 0.8, random_state = 55
)

x_train,x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                test_size = 0.2, shuffle = True)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


# 2. 모델구성 -> 통상적으로 밑으로 갈 수록 좋다고 하는 모델.
model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier() # ==> rf = RandomForestClassifier(), 'model'은 변수명 정의한 것. 

# 3. 훈련

model.fit(x_train,y_train)

#4. 평가, 예측

#result = model.evaluate(x_test,y_test)
result = model.score(x_test,y_test) # evaluate에서 나오는건 loss,acc였는데 score하면 acc바로 나옴. == model.score는 evaluate의 metrics에서 acc값과 같다. 
print('result : ', result)
#0.9666666666666667

# sklearn 나온 이후에 tensorflow 나왔다. 그래서 model.score보다 model.evaluate가 기능이 더 많다. 

y_predict = model.predict(x_test) # predict는 머신러닝과 동일하게 사용
# print(y_predict)

acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ',acc)

###########################
# Tensorflow
# acc : 1.0
###########################

# LinearSVC() 
# result :  1.0
# accuracy_score :  1.0

# SVC
# result :  1.0
# accuracy_score :  1.0

# KNeighborsClassifier
#result :  1.0
# accuracy_score :  1.0

# LogisticRegression
# result :  1.0
# accuracy_score :  1.0

# DecisionTreeClassifier
# result :  0.9444444444444444
# accuracy_score :  0.9444444444444444

# RandomForestClassifier
# result :  1.0
# accuracy_score :  1.0
