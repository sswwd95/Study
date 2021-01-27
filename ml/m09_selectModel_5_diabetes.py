from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_diabetes
import warnings

warnings.filterwarnings('ignore') # warning에 대해서 무시하겠다.

dataset = load_diabetes()
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
        # continue
        print(name,'은 없는 놈!')

'''
ARDRegression 의 정답률 :  0.5191329921161931
AdaBoostRegressor 의 정답률 :  0.469809173089612
BaggingRegressor 의 정답률 :  0.46912256043876355
BayesianRidge 의 정답률 :  0.5229971756095793
CCA 의 정답률 :  0.5125333896731441
DecisionTreeRegressor 의 정답률 :  -0.16718351419721
DummyRegressor 의 정답률 :  -0.008921922297815854
ElasticNet 의 정답률 :  0.0007444363916572216
ElasticNetCV 의 정답률 :  0.48061374963876846
ExtraTreeRegressor 의 정답률 :  -0.19710113985699418
ExtraTreesRegressor 의 정답률 :  0.5301328821530246
GammaRegressor 의 정답률 :  -0.0014301558021290184
GaussianProcessRegressor 의 정답률 :  -24.381234417653374
GeneralizedLinearRegressor 의 정답률 :  -0.0017326005670086353
GradientBoostingRegressor 의 정답률 :  0.40124580357617023
HistGradientBoostingRegressor 의 정답률 :  0.5038300819026051
HuberRegressor 의 정답률 :  0.5176273917367221
IsotonicRegression 은 없는 놈!
KNeighborsRegressor 의 정답률 :  0.5292709826835631
KernelRidge 의 정답률 :  -3.4547766894303233
Lars 의 정답률 :  0.5164196511816808
LarsCV 의 정답률 :  0.5205086417293971
Lasso 의 정답률 :  0.3942702522363706
LassoCV 의 정답률 :  0.5188308427255368
LassoLars 의 정답률 :  0.4215837430891125
LassoLarsCV 의 정답률 :  0.5190150804649276
LassoLarsIC 의 정답률 :  0.5178063240108697
LinearRegression 의 정답률 :  0.5164196511816804
LinearSVR 의 정답률 :  -0.27115844543772294
MLPRegressor 의 정답률 :  -3.012201503397324
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet 은 없는 놈!
MultiTaskElasticNetCV 은 없는 놈!
MultiTaskLasso 은 없는 놈!
MultiTaskLassoCV 은 없는 놈!
NuSVR 의 정답률 :  0.17872086335364923
OrthogonalMatchingPursuit 의 정답률 :  0.3206279007288464
OrthogonalMatchingPursuitCV 의 정답률 :  0.516295624897841
PLSCanonical 의 정답률 :  -1.626730286626418
PLSRegression 의 정답률 :  0.5188503094908998
PassiveAggressiveRegressor 의 정답률 :  0.511316099204606
PoissonRegressor 의 정답률 :  0.3816814218362221
RANSACRegressor 의 정답률 :  0.22215947838762906
RadiusNeighborsRegressor 의 정답률 :  -0.008921922297815854
RandomForestRegressor 의 정답률 :  0.47110520645773624
RegressorChain 은 없는 놈!
Ridge 의 정답률 :  0.4601036245070298
RidgeCV 의 정답률 :  0.5250902611436197
SGDRegressor 의 정답률 :  0.4433580885979892
SVR 의 정답률 :  0.19324326408908432
StackingRegressor 은 없는 놈!
TheilSenRegressor 의 정답률 :  0.5206748271586805
TransformedTargetRegressor 의 정답률 :  0.5164196511816804
TweedieRegressor 의 정답률 :  -0.0017326005670086353
VotingRegressor 은 없는 놈!
_SigmoidCalibration 은 없는 놈!
'''

############################
# Tensorflow
# R2 : 0.5094325786699055
############################

############################################
# 머신러닝
# ExtraTreesRegressor 의 정답률 :  0.5301328821530246
############################################

