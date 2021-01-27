'''
<인공지능의 겨울>
* 첫번째 암흑기(1974-1980)
1969년 마빈 민스키와 세이무어 페퍼트는 저서를 통해 퍼셉트론은 AND 또는 OR 같은 선형 분리가 가능한 문제는 가능하지만,  
선형(linear) 방식으로 데이터를 구분할 수 없는 XOR문제에는 적용할 수 없다는 것을 수학적으로 증명했다. 
이에 따라 미국방부 DARPA는 AI 연구자금을 2000만달러를 전격 중단하기에 이르렀습니다.
또한 영국의 라이트힐 경은 영국의회에 “폭발적인 조합증가(Combinational explosion)를 인공지능이 다룰(Intractability)수 없다” 라고 보고함으로써, 
사실상 인공지능에 대한 대규모 연구는 중단되어 다시 한번 암흑기에 접어들게 된다.
'''

from sklearn.svm import LinearSVC # 머신러닝 모델 중 하나. 선형모델
import numpy as np
from sklearn.metrics import accuracy_score # 회귀에서 r2와 같은 역할. 분류에서는 accuracy_score 사용

# 1. 데이터

# AND 데이터(모든 입력값이 1일 때만 1)
x_data = [[0,0],[1,0],[0,1],[1,1]]
y_data = [0,0,0,1]

# 2. 모델
model = LinearSVC()

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측

y_pred = model.predict(x_data)
print(x_data,'의 예측 결과 : ', y_pred)
# [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과 :  [0 0 0 1]

result = model.score(x_data, y_data)
print('model.score :', result)
# model.score : 1.0

acc = accuracy_score(y_data, y_pred)
print('accuracy_score : ',acc)
# accuracy_score :  1.0