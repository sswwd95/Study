# 실습
# 1. 상단모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성
# 최적의 r2값과 피처임포턴스 구할것

# 2. 위 쓰레드 값으로 selectfrommodel을 구해서 최적의 피처 갯수를 구할것

# 3. 위 피처 갯수로 데이터(피처)를 수정(삭제)해서
# 그리드 서치 또는 랜덤서치 적용하여 최적의 r2구할 것

# 1번값과 2번값 비교

from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
    GridSearchCV, RandomizedSearchCV

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66
)

xgb = XGBRegressor()

parameters = {
    'max_depth': [2, 4, 6, -1],
    'min_child_weight': [1, 2, 4, -1],
    'eta': [0.3, 0.1, 0.01, 0.5]
}

# 2. 모델구성
model = RandomizedSearchCV(xgb, param_distributions=parameters, cv=5)


model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2 : ', score)

# 이렇게 하면 fi 값들이 정렬되어 나온다!
thresholds = np.sort(model.best_estimator_.feature_importances_)
print(thresholds)
# R2 :  0.9328109624443419
# [0.0007204  0.00150525 0.01197547 0.01352609 0.01669537 0.02149532
#  0.02261044 0.0360474  0.05330402 0.05927434 0.07038534 0.29080436
#  0.40165624]

tmp = 0
tmp2 = [0, 0]


# 하나씩 포문돌린다
for thresh in thresholds:
    # selection = SelectFromModel(model, threshold = thresh, prefit = True)
    selection = SelectFromModel(
        model.best_estimator_, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs=13)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)

    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)
    if score > tmp:
        tmp = score
        tmp2[0] = thresh
        tmp2[1] = select_x_train.shape[1]

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %
          (thresh, select_x_train.shape[1], score*100))
    print('best score so far : ', tmp)
    print('best threshold :', tmp2[0])

print("++++++++++++++++++++++++++++++++++++++++++++++++++")

print(f'best threshold : {tmp2[0]}, n={tmp2[1]}')

selection = SelectFromModel(model.best_estimator_,
                            threshold=tmp2[0], prefit=True)

select_x_train = selection.transform(x_train)

selection_model = RandomizedSearchCV(xgb, parameters, cv=5)
selection_model.fit(select_x_train, y_train)

select_x_test = selection.transform(x_test)
y_predict = selection_model.predict(select_x_test)

score = r2_score(y_test, y_predict)

print("=======================================")
print(f'최종 r2 score : {score*100}%, n={tmp2[1]}일 때')


'''
(404, 13)
Thresh=0.002, n=13, R2: 92.21%
(404, 12)
Thresh=0.007, n=12, R2: 91.96%
(404, 11)
Thresh=0.009, n=11, R2: 92.03%
(404, 10)
Thresh=0.009, n=10, R2: 93.07%
(404, 9)
Thresh=0.011, n=9, R2: 93.34%
(404, 8)
Thresh=0.020, n=8, R2: 93.52%
(404, 7)
Thresh=0.021, n=7, R2: 92.86%
(404, 6)
Thresh=0.026, n=6, R2: 92.71%
(404, 5)
Thresh=0.027, n=5, R2: 91.74%
(404, 4)
Thresh=0.041, n=4, R2: 92.11%
(404, 3)
Thresh=0.052, n=3, R2: 92.52%
(404, 2)
Thresh=0.217, n=2, R2: 69.41%
(404, 1)
Thresh=0.560, n=1, R2: 44.98%
'''
# 최종 r2 score : 92.25967326472869%, n=7일 때
