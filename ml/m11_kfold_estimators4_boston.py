from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import accuracy_score,r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_boston
import warnings

warnings.filterwarnings('ignore') # warning에 대해서 무시하겠다.

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 55)
kfold = KFold(n_splits=5,shuffle=True)

allAlgorithms = all_estimators(type_filter='regressor')
for (name, algorithm) in allAlgorithms : 
    try:
        model = algorithm() 
        score = cross_val_score(model, x_train, y_train, cv=kfold)

        # model.fit(x_train,y_train)
        # y_pred = model.predict(x_test)
        print (name, '의 정답률 : ', score)
    except : 
        continue
        print(name,'은 없는 놈!')

'''
ARDRegression 의 정답률 :  [0.73502643 0.64740243 0.64150047 0.72324229 0.71168225]
AdaBoostRegressor 의 정답률 :  [0.83177335 0.9093433  0.76442959 0.87318448 0.880041  ]
BaggingRegressor 의 정답률 :  [0.91059384 0.88123825 0.79004111 0.78690192 0.81287205]
BayesianRidge 의 정답률 :  [0.69145585 0.58885158 0.581995   0.78629861 0.80338529]
CCA 의 정답률 :  [0.59850449 0.68661267 0.82686988 0.6142266  0.50843078]
DecisionTreeRegressor 의 정답률 :  [0.5090694  0.64025306 0.80811506 0.83730371 0.83435175]
DummyRegressor 의 정답률 :  [-0.00019138 -0.00014268 -0.00244962 -0.05119602 -0.06202379]
ElasticNet 의 정답률 :  [0.67002246 0.58878636 0.7111401  0.72745945 0.59429524]
ElasticNetCV 의 정답률 :  [0.75253722 0.5348901  0.65585511 0.67869235 0.6712591 ]
ExtraTreeRegressor 의 정답률 :  [0.71438492 0.79004696 0.79847775 0.69550798 0.69297419]
ExtraTreesRegressor 의 정답률 :  [0.93023939 0.72877276 0.87582468 0.87955677 0.92566924]
GammaRegressor 의 정답률 :  [-3.66040818e-05 -1.64235627e-02 -2.50490910e-02 -1.58220077e-02
 -1.06377266e-02]
GaussianProcessRegressor 의 정답률 :  [-4.56892885 -5.36957773 -7.35810069 -6.48535426 -5.31057533]
GeneralizedLinearRegressor 의 정답률 :  [0.68541635 0.66626495 0.70683436 0.57148097 0.57845391]
GradientBoostingRegressor 의 정답률 :  [0.89641448 0.83125648 0.92475438 0.90269581 0.86884468]
HistGradientBoostingRegressor 의 정답률 :  [0.87836698 0.90567743 0.85583623 0.87221465 0.70629587]
HuberRegressor 의 정답률 :  [0.44051091 0.62150984 0.59158756 0.72561529 0.57807869]
IsotonicRegression 의 정답률 :  [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 :  [0.31333887 0.37582859 0.5172915  0.60528025 0.69871526]
KernelRidge 의 정답률 :  [0.73785483 0.73557782 0.65826873 0.54281945 0.68220256]
Lars 의 정답률 :  [0.72990528 0.59692841 0.77391629 0.61082612 0.72418729]
LarsCV 의 정답률 :  [0.75266295 0.44049079 0.73033267 0.71335777 0.71246998]
Lasso 의 정답률 :  [0.65618426 0.63483237 0.72182067 0.57849825 0.65897341]
LassoCV 의 정답률 :  [0.75010595 0.71665255 0.56848683 0.66862651 0.63076142]
LassoLars 의 정답률 :  [-2.73873822e-03 -3.51656543e-03 -1.41288246e-05 -3.67757078e-04
 -8.84934896e-05]
LassoLarsCV 의 정답률 :  [0.78320484 0.68081168 0.72709465 0.69856654 0.6475636 ]
LassoLarsIC 의 정답률 :  [0.7561808  0.76025205 0.70549853 0.62968272 0.68561309]
LinearRegression 의 정답률 :  [0.66042675 0.61464028 0.80092103 0.75007784 0.71248103]
LinearSVR 의 정답률 :  [ 0.4140062  -1.74887303  0.45709334  0.66370556  0.60319329]
MLPRegressor 의 정답률 :  [0.52219656 0.69056549 0.5149806  0.49868164 0.79772284]
MultiTaskElasticNet 의 정답률 :  [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 :  [nan nan nan nan nan]
MultiTaskLasso 의 정답률 :  [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 :  [nan nan nan nan nan]
NuSVR 의 정답률 :  [0.21791658 0.13365927 0.31791795 0.30852942 0.18499667]
OrthogonalMatchingPursuit 의 정답률 :  [0.5719707  0.57895938 0.48064842 0.56281716 0.50634956]
OrthogonalMatchingPursuitCV 의 정답률 :  [0.73873413 0.54813045 0.6311001  0.61379428 0.76088666]
PLSCanonical 의 정답률 :  [-1.58689493 -2.93690514 -1.79769894 -2.61317807 -2.04563368]
PLSRegression 의 정답률 :  [0.67885811 0.74201835 0.71201912 0.52657585 0.69526235]
PassiveAggressiveRegressor 의 정답률 :  [ 0.17316486  0.28236155 -0.26841593  0.09325933  0.26232448]
PoissonRegressor 의 정답률 :  [0.67009352 0.76462006 0.75949795 0.78117325 0.77847164]
RANSACRegressor 의 정답률 :  [ 0.42986171  0.60839859  0.68063969  0.80026322 -3.18488443]
RandomForestRegressor 의 정답률 :  [0.87475114 0.9054655  0.85013832 0.8764199  0.88707046]
Ridge 의 정답률 :  [0.63506074 0.61312398 0.74358471 0.76444111 0.79174326]
RidgeCV 의 정답률 :  [0.68925478 0.69042073 0.7989097  0.69262128 0.71334318]
SGDRegressor 의 정답률 :  [-1.58224559e+25 -7.99876694e+25 -7.25947445e+26 -2.78766920e+25
 -1.06279738e+27]
SVR 의 정답률 :  [0.28431703 0.22205315 0.12559106 0.21189013 0.14097859]
TheilSenRegressor 의 정답률 :  [0.72851976 0.79060469 0.72468257 0.76047015 0.33225589]
TransformedTargetRegressor 의 정답률 :  [0.73100047 0.63428205 0.73215164 0.5835577  0.78981645]        
TweedieRegressor 의 정답률 :  [0.60321566 0.66277113 0.67543758 0.62036881 0.69511569]
_SigmoidCalibration 의 정답률 :  [nan nan nan nan nan]
'''

###########################
# Tensorflow
# LSTM 
# acc : 0.9824561476707458
###########################
############################################
# 머신러닝
# RidgeClassifier 의 정답률 :  0.9824561403508771
############################################