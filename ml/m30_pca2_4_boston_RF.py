
# rf로 모델링하기

import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
# deomposition 분해
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
import warnings
warnings.filterwarnings('ignore')


datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)
'''
pca = PCA(n_components=10)
x2 = pca.fit_transform(x) # fit과 transform 합친 것
print(x2)
print(x2.shape) #(442, 7) 컬럼의 수가 재구성

pca_EVR = pca.explained_variance_ratio_ # 변화율
print(pca_EVR) #[0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192 0.05365605]
print(sum(pca_EVR)) 
# 7개 : 0.9479436357350414 
# 8개 : 0.9913119559917797
# 9개 : 0.9991439470098977
# 10개 : 1.0
# 몇 개가 좋은지 어떻게 알까? 모델 돌려보면 알 수 있다. 통상적으로 95% 이면 모델에서 성능 비슷하게 나온다. 
'''

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
# cunsum의 작은 것 부터 하나씩 더해준다. 함수는 주어진 축에서 배열 요소의 누적 합계를 계산하려는 경우에 사용된다.
print(cumsum)
# [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395 1.        ]

d = np.argmax(cumsum>=0.95)+1
print('cumsum >=0.95', cumsum >=0.95)
print('선택할 차원 수 :', d)

# cumsum >=0.95 [False False False False False False False  True  True  True]
# d :  8
 
#2. 모델
model = RandomForestRegressor()

# 3.훈련
model.fit(x, y)

# 4. 평가, 예측
acc = model.score(x, y)
print('acc : ', acc)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

# acc :  0.9828073421427419