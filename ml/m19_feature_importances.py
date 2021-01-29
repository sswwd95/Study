from sklearn.tree import DecisionTreeClassifier # tree구조의 모델
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

# [0.         0.         0.96990618 0.03009382] -> 피처의 순번대로 출력.피처에서 2개 빼도 똑같다는 것. 피처가 많다고 좋은 것이 아니다. 
# acc :  0.9666666666666667