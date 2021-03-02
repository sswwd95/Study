import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# 2. 모델
model = Sequential()
model.add(Dense(4,input_dim = 1))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

print(model.weights)

'''
[<tf.Variable 'dense/kernel:0' shape=(1, 4) dtype=float32, numpy=
array([[-0.4092027 ,  0.08582067,  0.5063828 ,  0.70251703]],dtype=float32)>, -> 다음 레이어로 전달되는 weight값
      <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>, => bias값

<tf.Variable 'dense_1/kernel:0' shape=(4, 3) dtype=float32, numpy=
array([[ 0.46398675, -0.6593244 ,  0.33257294],
       [-0.21667832, -0.25190324, -0.8438519 ],
       [ 0.6613163 , -0.69617665, -0.75165385],
       [ 0.6566651 , -0.61016285, -0.6634722 ]], dtype=float32)>, 
       <tf.Variable 'dense_1/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>,
       
<tf.Variable 'dense_2/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[ 0.4080112 , -0.4169277 ],
       [-0.46522093, -0.8379577 ],
       [ 0.24277973,  0.06909323]], dtype=float32)>,
       <tf.Variable 'dense_2/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>,

<tf.Variable 'dense_3/kernel:0' shape=(2, 1) dtype=float32, numpy=
array([[0.6582593],[1.3025848]], dtype=float32)>, 
<tf.Variable 'dense_3/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>
'''

print(model.trainable_weights) # 훈련시키는 weight값

# 전이학습하게 된 것은 non-trainable (훈련을 시킬 필요가 없다)

print(len(model.weights)) #8(weight , bias 모두 더한 값. 1개의 layer에 1개의 weight 값, 1개의 bias 값 => 총 layer 4개 있으니 결과는 8)
print(len(model.trainable_weights)) #8


