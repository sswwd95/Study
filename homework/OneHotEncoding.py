#  data (3,6,5,4,2)

import numpy as np
from numpy import array

data = array([3,6,5,4,2])

from tensorflow.keras.utils import to_categorical 
data = to_categorical(data)

print(data.shape)  # (5, 7)

print(data)
# [[0. 0. 0. 1. 0. 0. 0.]   # 0부터 3번째 : 3
#  [0. 0. 0. 0. 0. 0. 1.]   #  ''   6번째 : 6
#  [0. 0. 0. 0. 0. 1. 0.]   #  ''   5번째 : 5
#  [0. 0. 0. 0. 1. 0. 0.]   #  ''   4번째 : 4
#  [0. 0. 1. 0. 0. 0. 0.]]  #  ''   2번째 : 2

# to_categorical은 0부터 시작해서 주어진 값에 1을 줌. 주어진 값에 1을 부여하기 위해 0을 배열해서 (행, 열)이 같지 않음.

#####################################################

import numpy as np
from numpy import array

data = array([3,6,5,4,2])

data = data.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(data)
data= enc.transform(data).toarray()

print(data.shape)  # (5, 5)

print(data)

# [[0. 1. 0. 0. 0.]   # 2번째 : 3
#  [0. 0. 0. 0. 1.]   # 5번째 : 6
#  [0. 0. 0. 1. 0.]   # 4번째 : 5
#  [0. 0. 1. 0. 0.]   # 3번째 : 4
#  [1. 0. 0. 0. 0.]]  # 1번째 : 2

# OneHotEncoding은 1부터 시작해서 작은 숫자를 1로 설정한다. 
# 위의 예시에서 2가 제일 크기가 작으니 1번...6이 제일 숫자가 크니까 5번째에 1.
# (행, 열)크기가 같음.


