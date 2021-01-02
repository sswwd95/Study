# 실습 : 19_1,2,3,4,5 Early stopping까지 총 2개의 파일 완성하기
# 1. Early stopping을 적용하지 않은 최고의 모델
# 2. Early stopping을 적용한 최고의 모델
 
from tensorflow.keras.datasets import boston_housing

# 이걸로 만들어라! sklearn이 아니라 tensorflow에서 가져온거기때문에 x,y 나누지 않는다. 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터

(train_X, train_Y), (test_X, test_Y) = boston_housing.load_data()

print(train_X.shape)  # (404,13)
print(train_Y.shape)  # (404, )

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    train_X, train_Y, train_size = 0.8, random_state = 66
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성

model.
