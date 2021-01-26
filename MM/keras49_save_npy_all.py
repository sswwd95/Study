# (sklearn) boston, diabetes, cancer, iris, wine
# (keras) mnist, fashion, cifar10, cifar100 
# sklearn과 중복되면 sklearn으로 한다.

# 이름 바꿔주는 이유 :  파일 안에서 이름 중복이라서. 각 파일로 만들면 상관없음.
# numpy save하는 이유? csv로 받으면 엑셀파일로 받아서 너무 느리다. numpy로 저장하면 훨씬 빠르게 불러올 수 있다. 

from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer, load_iris, load_wine
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np

# 1. boston
boston_dataset = load_boston()
boston_x = boston_dataset.data
boston_y = boston_dataset.target

np.save('../data/npy/boston_x.npy', arr = boston_x)
np.save('../data/npy/boston_y.npy', arr = boston_y)

# 2. diabetes
diabetes_dataset = load_diabetes()
diabetes_x = diabetes_dataset.data
diabetes_y = diabetes_dataset.target

np.save('../data/npy/diabetes_x.npy', arr = diabetes_x)
np.save('../data/npy/diabetes_y.npy', arr = diabetes_y)

# 3. cancer
cancer_dataset = load_breast_cancer()
cancer_x = cancer_dataset.data
cancer_y = cancer_dataset.target

np.save('../data/npy/cancer_x.npy', arr = cancer_x)
np.save('../data/npy/./data/cancer_y.npy', arr = cancer_y)

# 4. iris
iris_dataset = load_iris()
iris_x = iris_dataset.data
iris_y = iris_dataset.target

np.save('../data/npy/iris_x.npy', arr = iris_x)
np.save('../data/npy/iris_y.npy', arr = iris_y)

# 5. wine
wine_dataset = load_wine()
wine_x = wine_dataset.data
wine_y = wine_dataset.target

np.save('../data/npy/wine_x.npy', arr = wine_x)
np.save('../data/npy/wine_y.npy', arr = wine_y)

# 6. mnist
(m_x_train,m_y_train), (m_x_test, m_y_test) = mnist.load_data()
np.save('../data/npy/mnist_x_train.npy', arr=m_x_train)
np.save('../data/npy/mnist_x_test.npy', arr=m_x_test)
np.save('../data/npy/mnist_y_train.npy', arr=m_y_train)
np.save('../data/npy/mnist_y_test.npy', arr=m_y_test)

# 7. fashion_mnist
(f_x_train,f_y_train), (f_x_test, f_y_test) = fashion_mnist.load_data()
np.save('../data/npy/f_x_train.npy', arr=f_x_train)
np.save('../data/npy/f_x_test.npy', arr=f_x_test)
np.save('../data/npy/f_y_train.npy', arr=f_y_train)
np.save('../data/npy/f_y_test.npy', arr=f_y_test)

# 8. cifar10
(c_x_train,c_y_train), (c_x_test, c_y_test) = cifar10.load_data()
np.save('../data/npy/c_x_train.npy', arr=c_x_train)
np.save('../data/npy/c_x_test.npy', arr=c_x_test)
np.save('../data/npy/c_y_train.npy', arr=c_y_train)
np.save('../data/npy/c_y_test.npy', arr=c_y_test)

# 9. cifar100
(i_x_train,i_y_train), (i_x_test, i_y_test) = cifar100.load_data()
np.save('../data/npy/i_x_train.npy', arr=i_x_train)
np.save('../data/npy/i_x_test.npy', arr=i_x_test)
np.save('../data/npy/i_y_train.npy', arr=i_y_train)
np.save('../data/npy/i_y_test.npy', arr=i_y_test)