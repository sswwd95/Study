# 컬럼이 많을 경우에 다 돌리는 건 자원낭비
# 차원축소
# 95%의 특성만 가져도 loss, acc 뺄 수 있다고 생각하면 pca써서 차원 축소하기.
# 압축을 했을 때 몇개를 쓰면 좋은지 알 수 있다. 
# feature importance쓰면 기본 데이터 변형없지만 pca쓰면 변형된다.하지만 전처리 개념으로 생각해서 변형되어도 상관없다. 
# pca하면 전처리 해도되고 안해도 된다. 만약 한다면 전처리 한 후에 pca적용

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
# deomposition 분해

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(442, 10) (442,)

pca = PCA(n_components=7) # 축소하고 싶은 만큼 지정
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