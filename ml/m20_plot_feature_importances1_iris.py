# 중요하지 않은 피처라고 나와도 Decisiontree 모델을 신뢰할 수 없기 때문에 뺄 수 없다. 
# 상관관계도와 비교해서 더 맞다고 생각하는 걸 선택. 

# 중요하지 않은 피처 그래프로 확인하기

from sklearn.tree import DecisionTreeClassifier 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
)

# 2. 모델
model=DecisionTreeClassifier(max_depth=4) # 깊이 4

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가,예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print('acc : ', acc)

# [0.         0.         0.96990618 0.03009382] 
# acc :  0.9666666666666667

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model) : 
    n_features = dataset.data.shape[1] #datset shape (150,4)
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel('feature importances')
    plt.ylabel('features')
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

# sepal width, sepal length = 중요도 낮은 피처