'''
import numpy as np
from sklearn.covariance import EllipticEnvelope
# aaa = np.array([1,2,-10000, 3,4,6,7,8,90,100,5000])
# print(aaa.shape) #(11,)

aaa = np.array([[1,2,-10000, 3,4,6,7,8,90,100,5000]])
print(aaa.shape) #(1,11)
aaa = np.transpose(aaa)
print(aaa.shape)#(11, 1)

outlier = EllipticEnvelope(contamination=.2) 
# 20%의 오염도를 찾는다
outlier.fit(aaa)
print(outlier.predict(aaa))
# [ 1  1 -1  1  1  1  1  1  1  1 -1]

outlier = EllipticEnvelope(contamination=.1)
 # 10%의 오염도를 찾는다
outlier.fit(aaa)
print(outlier.predict(aaa))
# [ 1  1 -1  1  1  1  1  1  1  1  1]

# 통상 0.1 아래로 잡는다. 0.3으로 잡으면 ? 그건 outlier가 아니라 실제 데이터일 확률이 높다
'''

import numpy as np
from sklearn.covariance import EllipticEnvelope
# aaa = np.array([[1,2,-10000, 3,4,6,7,8,90,100,5000]])
# [ 1  1 -1  1  1  1  1  1  1  1 -1]

# aaa = np.array([[1000,2000,3,4000,5000,6000,7000,8,9000,10000,1001]])
# [ 1  1  1  1  1  1  1  1 -1 -1  1]

aaa = np.array([[1,2,-10000, 3,4,6,7,8,90,100,5000],
                [1000,2000,3,4000,5000,6000,7000,8,9000,10000,1001]])
                
print(aaa.shape) #(2,11)
aaa = np.transpose(aaa)
print(aaa.shape)#(11, 2)

outlier = EllipticEnvelope(contamination=.2)
outlier.fit(aaa)
print(outlier.predict(aaa))
#[ 1  1 -1  1  1  1  1  1  1  1 -1]

# outlier는 전처리 전에 작업해야한다.

