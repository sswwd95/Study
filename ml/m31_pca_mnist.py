import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier
from tensorflow.keras.datasets import mnist


(x_train,_), (x_test,_) = mnist.load_data()
# _ 는 y 값을 지정하지 않는다는 것
x = np.append(x_train, x_test, axis=0)
print(x.shape) #(70000, 28, 28)

# ->  2차원으로 reshape
x = x.reshape(-1,x.shape[1]*x.shape[2])
print(x.shape) #(70000, 784)

# 실습
# pca를 통해 0.95이상이 몇개인지?

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)
'''
d = np.argmax(cumsum>=0.95)+1
print('cumsum >=0.95', cumsum >=0.95)
print('d : ', d) #d :  154
'''
# d :  331 (0.99)
d = np.argmax(cumsum>=1.0)+1
print('cumsum >=0.95', cumsum >=1.0)
print('d : ', d) 
# d :  713
'''
pca = PCA(n_components=154)
x2 = pca.fit_transform(x)

x_train, x_test= train_test_split(x2, train_size=0.8, random_state=77 )

model = Pipeline([('scaler', MinMaxScaler()),('model', XGBClassifier())])

model.fit(x_train)

acc = model.score(x_test)
print('acc : ', acc)

'''
