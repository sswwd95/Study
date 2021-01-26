import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1, activation='linear')) 
model.add(Dense(6, activation='linear'))      
model.add(Dense(4, name = 'aa'))

model.add(Dense(1))

model.summary()

# layer을 만들 때 'name'이란 것에 대해 확인하고 설명할 것
# name을 반드시 써야할 때가 있다. 그때를 말해라.
'''
name을 layer에서 고유한 선택적 문자열이다. 같은 이름을 두 번 재사용은 안되며 제공되지 않으면 자동으로 생성된다

model.add(Dense(4, name = 'new name')) 을 입력하고 돌리면 
ValueError: 'new name/' is not a valid scope name 이라고 뜨면서 에러가 난다. 
기존의 읽기 전용 property와 충돌하기 때문에 설정할 수 없다. 
'''

#######################################################################################################
# 앙상블의 summary 이해하기

'''
from tensorflow.keras.layers import Dense, Input

# 모델1
input1 = Input(shape=(3,))
dense1 = Dense(10,activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)
# output1 = Dense(3)(dense1)

# 모델2
input2 = Input(shape=(3,))
dense2 = Dense(10,activation='relu')(input2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)

# 모델 병합 / concatenate
from tensorflow.keras.layers import concatenate, Concatenate

# merge = 합치다
merge1 = concatenate([dense1, dense2]) # 제일 끝의 dense 변수명 넣기
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)

# 모델 분기1
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

# 모델 분기2
output2 = Dense(15)(middle1)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(3)(output2)

__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_2 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 10)           40          input_2[0][0]
__________________________________________________________________________________________________
input_1 (InputLayer)            [(None, 3)]          0
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 5)            55          dense_2[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 10)           40          input_1[0][0]
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 5)            30          dense_3[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 5)            55          dense[0][0]
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 5)            30          dense_4[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 10)           0           dense_1[0][0]
                                                                 dense_5[0][0]
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 30)           330         concatenate[0][0]
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 10)           310         dense_6[0][0]
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 10)           110         dense_7[0][0]
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 15)           165         dense_8[0][0]
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 30)           330         dense_8[0][0]
__________________________________________________________________________________________________
dense_13 (Dense)                (None, 7)            112         dense_12[0][0]
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 7)            217         dense_9[0][0]
__________________________________________________________________________________________________
dense_14 (Dense)                (None, 7)            56          dense_13[0][0]
__________________________________________________________________________________________________
dense_11 (Dense)                (None, 3)            24          dense_10[0][0]
__________________________________________________________________________________________________
dense_15 (Dense)                (None, 3)            24          dense_14[0][0]
==================================================================================================
Total params: 1,928
Trainable params: 1,928
Non-trainable params: 0
__________________________________________________________________________________________________

ensemble에서는 함수를 쓰는데 seqeuntial에서는 순서대로 나와있어서 눈으로 바로 계산하기 편했지만 함수모델에서는 순서가 섞여 있다.
inputlayer에서는 파라미터가 0인데 받는 인수가 없어 파라미터 값이 0이다. 
connected to는 인풋을 받는 레이어 표시며 어디에 연결된건지 보면서 계산할 수 있다. 
concatenate는 레이어들을 결합해서 output하기 때문에 inputlayer와 마찬가지로 파라미터가 0이다. 
그 외 중간 계산하는건 Seqeuntial에서 공부한 것과 같이 input값에 +b값(+1)을 하고 노드를 곱하면 파라미터를 계산할 수 있다. 

'''