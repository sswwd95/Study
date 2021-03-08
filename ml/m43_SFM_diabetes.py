# 당뇨병 만들어보기
# 0.5이상!

# 실습
# 1. 상단모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성
# 최적의 r2값과 피처임포턴스 구할것

# 2. 위 쓰레드 값으로 selectfrommodel을 구해서 최적의 피처 갯수를 구할것

# 3. 위 피처 갯수로 데이터(피처)를 수정(삭제)해서 
# 그리드 서치 또는 랜덤서치 적용하여 최적의 r2구할 것

# 1번값과 2번값 비교

from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier,XGBRegressor, plot_importance
from sklearn.pipeline import Pipeline, make_pipeline 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score,\
     GridSearchCV, RandomizedSearchCV

x, y = load_diabetes(return_X_y= True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = True, random_state = 66
)

xgb = XGBRegressor()

parameters = {
    'max_depth' : [2,4,6,-1],
    'min_child_weight' : [1,2,4,-1],
    'eta' : [0.3,0.1,0.01,0.5]
}

# 2. 모델구성
model = RandomizedSearchCV(xgb, param_distributions=parameters, cv=5)


model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2 : ', score)

thresholds = np.sort(model.best_estimator_.feature_importances_) # 이렇게 하면 fi 값들이 정렬되어 나온다!
print(thresholds)
# R2 :  0.9328109624443419
# [0.0007204  0.00150525 0.01197547 0.01352609 0.01669537 0.02149532
#  0.02261044 0.0360474  0.05330402 0.05927434 0.07038534 0.29080436
#  0.40165624]

tmp = 0
tmp2 = [0,0]


# 하나씩 포문돌린다
for thresh in thresholds:
    # selection = SelectFromModel(model, threshold = thresh, prefit = True)
    selection = SelectFromModel(model.best_estimator_, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs = 30)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)

    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)
    if score>tmp:
        tmp = score
        tmp2[0] = thresh
        tmp2[1] = select_x_train.shape[1]

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100))
    print ('best score so far : ', tmp)
    print ('best threshold :',tmp2[0])

print("++++++++++++++++++++++++++++++++++++++++++++++++++")

print(f'best threshold : {tmp2[0]}, n={tmp2[1]}')

selection = SelectFromModel(model.best_estimator_, threshold=tmp2[0], prefit=True)

select_x_train = selection.transform(x_train)

selection_model = RandomizedSearchCV(xgb, parameters, cv=5)
selection_model.fit(select_x_train, y_train)

select_x_test = selection.transform(x_test)
y_predict = selection_model.predict(select_x_test)

score = r2_score(y_test, y_predict)

print("=======================================")
print(f'최종 r2 score : {score*100}%, n={tmp2[1]}일 때')

'''
R2 :  0.26747781040918406
[0.03441202 0.04032777 0.04686696 0.05166136 0.05288106 0.05692902
 0.06330046 0.12999505 0.18247978 0.34114653]
(353, 10)
Thresh=0.034, n=10, R2: 23.80%
best score so far :  0.23802704693460175
best threshold : 0.034412015
(353, 9)
Thresh=0.040, n=9, R2: 20.74%
best score so far :  0.23802704693460175
best threshold : 0.034412015
(353, 8)
Thresh=0.047, n=8, R2: 22.44%
best score so far :  0.23802704693460175
best threshold : 0.034412015
(353, 7)
Thresh=0.052, n=7, R2: 22.62%
best score so far :  0.23802704693460175
best threshold : 0.034412015
(353, 6)
Thresh=0.053, n=6, R2: 28.18%
best score so far :  0.28175346393016043
best threshold : 0.052881062
(353, 5)
Thresh=0.057, n=5, R2: 27.48%
best score so far :  0.28175346393016043
best threshold : 0.052881062
(353, 4)
Thresh=0.063, n=4, R2: 15.77%
best score so far :  0.28175346393016043
best threshold : 0.052881062
(353, 3)
Thresh=0.130, n=3, R2: 29.43%
best score so far :  0.2942964622166685
best threshold : 0.12999505
(353, 2)
Thresh=0.182, n=2, R2: 12.78%
best score so far :  0.2942964622166685
best threshold : 0.12999505
(353, 1)
Thresh=0.341, n=1, R2: 2.56%
best score so far :  0.2942964622166685
best threshold : 0.12999505
'''

# 최종 r2 score : 29.383283011201122%, n=3일 때