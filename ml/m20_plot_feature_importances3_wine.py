
from sklearn.tree import DecisionTreeClassifier 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_wine()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size=0.8, random_state=44
)

print(dataset.data.shape) #(178, 13)

# 2. 모델
model=DecisionTreeClassifier(max_depth=4) # 깊이 4

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
[0.         0.         0.         0.         0.         0.
 0.17679511 0.         0.         0.0179565  0.05577403 0.32933594
 0.42013842]
acc :  0.9166666666666666
['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 
'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 
'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'] 
'''  