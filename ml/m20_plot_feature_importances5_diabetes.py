
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
)

print(dataset.data.shape) #(442, 10)

# 2. 모델
model=DecisionTreeRegressor(max_depth=4) # 깊이 4

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가,예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print('acc : ', acc)

print(dataset.feature_names)

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
'''
[2.96958480e-02 2.16063823e-04 3.19268150e-01 0.00000000e+00
 1.83192445e-02 6.06279792e-02 0.00000000e+00 0.00000000e+00
 5.71872715e-01 0.00000000e+00]
acc :  0.2979755178635194
['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
'''  