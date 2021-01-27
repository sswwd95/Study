from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_boston
import warnings

warnings.filterwarnings('ignore') # warning에 대해서 무시하겠다.

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 55)

allAlgorithms = all_estimators(type_filter='regressor')
for (name, algorithm) in allAlgorithms : 
    try:
        model = algorithm() 

        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        print (name, '의 정답률 : ', r2_score(y_test,y_pred))
    except : 
        continue
        print(name,'은 없는 놈!')

'''
ARDRegression 의 정답률 :  0.7329622409161002
AdaBoostRegressor 의 정답률 :  0.8126785559287198
BaggingRegressor 의 정답률 :  0.8309289980417622
BayesianRidge 의 정답률 :  0.7568332174742188
CCA 의 정답률 :  0.6831383719037142
DecisionTreeRegressor 의 정답률 :  0.7245259436403003
DummyRegressor 의 정답률 :  -0.00644980499348935
ElasticNet 의 정답률 :  0.6998402634697864
ElasticNetCV 의 정답률 :  0.6779472360429024
ExtraTreeRegressor 의 정답률 :  0.11421900332844248
ExtraTreesRegressor 의 정답률 :  0.8478363985434756
GammaRegressor 의 정답률 :  -0.006449804993488906
GaussianProcessRegressor 의 정답률 :  -7.290865465953011
GeneralizedLinearRegressor 의 정답률 :  0.6601633755371181
GradientBoostingRegressor 의 정답률 :  0.8504673208570414
HistGradientBoostingRegressor 의 정답률 :  0.8677380412557965
HuberRegressor 의 정답률 :  0.707923024006128
KNeighborsRegressor 의 정답률 :  0.4874747847823836
KernelRidge 의 정답률 :  0.7501793217352599
Lars 의 정답률 :  0.7516965713967576
LarsCV 의 정답률 :  0.7566339935983615
Lasso 의 정답률 :  0.6946469090206508
LassoCV 의 정답률 :  0.7169685845515987
LassoLars 의 정답률 :  -0.00644980499348935
LassoLarsCV 의 정답률 :  0.7520549524652642
LassoLarsIC 의 정답률 :  0.7578199652426573
LinearRegression 의 정답률 :  0.7516965713967575
LinearSVR 의 정답률 :  -4.0087925075767945
MLPRegressor 의 정답률 :  0.5382311992401534
NuSVR 의 정답률 :  0.19926879995580882
OrthogonalMatchingPursuit 의 정답률 :  0.5361391650010913
OrthogonalMatchingPursuitCV 의 정답률 :  0.6973363625253238
PLSCanonical 의 정답률 :  -3.073802756930979
PLSRegression 의 정답률 :  0.7448546487212941
PassiveAggressiveRegressor 의 정답률 :  -0.2088007728189658
PoissonRegressor 의 정답률 :  0.766664755267804
RANSACRegressor 의 정답률 :  0.4222547523571344
RandomForestRegressor 의 정답률 :  0.8569968722508661
Ridge 의 정답률 :  0.7590919263795365
RidgeCV 의 정답률 :  0.7533367570599347
SGDRegressor 의 정답률 :  -1.0559964772652362e+27
SVR 의 정답률 :  0.15617931436054866
TheilSenRegressor 의 정답률 :  0.7264747835948473
TransformedTargetRegressor 의 정답률 :  0.7516965713967575
TweedieRegressor 의 정답률 :  0.6601633755371181
'''

############################
# Tensorflow
# R2 :  0.8732867386831191
############################

############################################
# 머신러닝
# HistGradientBoostingRegressor 의 정답률 :  0.8677380412557965
############################################