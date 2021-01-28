from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import accuracy_score,r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_diabetes
import warnings

warnings.filterwarnings('ignore') # warning에 대해서 무시하겠다.

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 55)
kfold = KFold(n_splits=5,shuffle=True)

allAlgorithms = all_estimators(type_filter='regressor')
for (name, algorithm) in allAlgorithms : 
    try:
        model = algorithm() 
        scores = cross_val_score(model, x_train, y_train, cv=kfold)

        # model.fit(x_train,y_train)
        # y_pred = model.predict(x_test)
        print (name, '의 정답률 : \n',scores)
    except : 
        continue
        print(name,'은 없는 놈!')

'''
ARDRegression 의 정답률 : 
 [0.42215286 0.46135974 0.57933602 0.52894471 0.34892082]
AdaBoostRegressor 의 정답률 : 
 [0.50411362 0.45504852 0.44149338 0.34113366 0.37752413]
BaggingRegressor 의 정답률 : 
 [0.42692773 0.47046964 0.45582137 0.17824111 0.2184218 ]
BayesianRidge 의 정답률 : 
 [0.42363495 0.42184718 0.49177779 0.54871307 0.40309081]
CCA 의 정답률 :
 [0.45545857 0.5071499  0.01628282 0.52776109 0.39037405]
DecisionTreeRegressor 의 정답률 : 
 [-0.11285808 -0.3862973   0.03725464 -0.28313352 -0.05102728]
DummyRegressor 의 정답률 :
 [-3.36596869e-02 -3.60130240e-03 -2.69433882e-03 -3.51310623e-03
 -1.48843220e-06]
ElasticNet 의 정답률 :
 [ 0.00657595 -0.02957113  0.00773063 -0.00488729 -0.10739396]
ElasticNetCV 의 정답률 : 
 [0.48804047 0.44646988 0.34383429 0.46336672 0.38152745]
ExtraTreeRegressor 의 정답률 :
 [-0.05360043 -0.31219054  0.3487055  -0.17490262 -0.29081325]
ExtraTreesRegressor 의 정답률 : 
 [0.34631896 0.24409024 0.39604588 0.42510824 0.56711512]
GammaRegressor 의 정답률 :
 [-0.00361858  0.00512435 -0.00500604 -0.00212881 -0.00520954]
GaussianProcessRegressor 의 정답률 : 
 [-33.04856114 -10.98986927  -8.26765571 -30.64054548  -3.71974106]
GeneralizedLinearRegressor 의 정답률 : 
 [-0.24572096 -0.00666866 -0.05447553 -0.02211616 -0.01401547]
GradientBoostingRegressor 의 정답률 : 
 [0.40739315 0.43197412 0.54285497 0.4518927  0.34007758]
HistGradientBoostingRegressor 의 정답률 : 
 [0.32444699 0.56427333 0.27720314 0.31227519 0.29290643]
HuberRegressor 의 정답률 : 
 [0.46435112 0.34266883 0.47200941 0.51526828 0.5182311 ]
IsotonicRegression 의 정답률 :
 [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 : 
 [0.40799799 0.20078526 0.19010455 0.26805825 0.24153407]
KernelRidge 의 정답률 :
 [-3.63608875 -3.87452171 -3.66966255 -3.62277925 -3.09326566]
Lars 의 정답률 : 
 [ 0.43183362 -0.12588013  0.39442188  0.5474299   0.55769188]
LarsCV 의 정답률 : 
 [ 0.24669686  0.49478049 -1.98914457  0.50760086  0.51447726]
Lasso 의 정답률 :
 [0.3172636  0.33774596 0.35697563 0.29991416 0.3278536 ]
LassoCV 의 정답률 : 
 [0.40262046 0.52026876 0.49853509 0.52831188 0.34506296]
LassoLars 의 정답률 :
 [0.33631785 0.39156221 0.45809906 0.33296535 0.381588  ]
LassoLarsCV 의 정답률 : 
 [0.41636692 0.5670046  0.44468844 0.54583099 0.32419923]
LassoLarsIC 의 정답률 :
 [0.48238403 0.37961166 0.59058572 0.47576912 0.39806349]
LinearRegression 의 정답률 : 
 [0.58324568 0.48095727 0.39544834 0.39346571 0.4795732 ]
LinearSVR 의 정답률 :
 [-0.32701865 -0.30706721 -0.82694589 -0.53300717 -0.38278835]
MLPRegressor 의 정답률 : 
 [-2.74526741 -2.56247183 -3.36045023 -3.38252141 -3.2334077 ]
MultiTaskElasticNet 의 정답률 :
 [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 :
 [nan nan nan nan nan]
MultiTaskLasso 의 정답률 : 
 [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 :
 [nan nan nan nan nan]
NuSVR 의 정답률 : 
 [0.13443327 0.11104764 0.14449176 0.13190374 0.12254486]
OrthogonalMatchingPursuit 의 정답률 :
 [0.47448441 0.47853326 0.10621588 0.29764081 0.23919968]
OrthogonalMatchingPursuitCV 의 정답률 : 
 [0.5025964  0.44880875 0.34138993 0.48015937 0.53204053]
PLSCanonical 의 정답률 : 
 [-1.83077455 -1.16758842 -1.62046786 -0.65024014 -0.66346575]
PLSRegression 의 정답률 :
 [0.48904762 0.44015909 0.51771036 0.49154671 0.43051689]
PassiveAggressiveRegressor 의 정답률 : 
 [0.35258415 0.30097348 0.45857011 0.48470359 0.55451613]
PoissonRegressor 의 정답률 : 
 [0.36596058 0.32702264 0.28066353 0.28643209 0.30364263]
RANSACRegressor 의 정답률 : 
 [-0.07362858 -0.29346248  0.13088738  0.13795297 -0.00097344]
RadiusNeighborsRegressor 의 정답률 :
 [-8.24236253e-04 -3.01762713e-05 -2.69710952e-03 -1.94572039e-02
 -4.39729627e-02]
RandomForestRegressor 의 정답률 : 
 [0.44832045 0.35600367 0.36707003 0.39891052 0.44046377]
Ridge 의 정답률 :
 [0.40077951 0.3944835  0.38836956 0.38436962 0.35612233]
RidgeCV 의 정답률 :
 [0.54224354 0.41266714 0.46321388 0.35174468 0.54043806]
SGDRegressor 의 정답률 : 
 [0.44892697 0.38233322 0.30009837 0.3800727  0.37318225]
SVR 의 정답률 : 
 [-0.03882382  0.11196066  0.14700817  0.16188431  0.14884886]
TheilSenRegressor 의 정답률 : 
 [0.31311734 0.48955177 0.49671444 0.45599181 0.42569962]
TransformedTargetRegressor 의 정답률 :
 [0.4281545  0.3863531  0.55725489 0.41597544 0.5040112 ]
TweedieRegressor 의 정답률 :
 [ 0.00024276 -0.00452602 -0.01116695  0.00471251 -0.07469684]
_SigmoidCalibration 의 정답률 :
 [nan nan nan nan nan]
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