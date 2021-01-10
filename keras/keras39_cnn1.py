
'''
Conv2D(32, (5, 5), padding='valid', input_shape=(28, 28, 1), activation='relu')

- 첫번째 인자 : 컨볼루션 필터의 수 입니다. output space(layer의 node) 
- 두번째 인자 : 컨볼루션 커널의 (행, 열) 입니다.
- padding : 경계 처리 방법을 정의합니다.
          ‘valid’ : 유효한 영역만 출력이 됩니다. 따라서 출력 이미지 사이즈는 입력 사이즈보다 작습니다.
          ‘same’ : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일합니다.
- input_shape : 샘플 수를 제외한 입력 형태를 정의 합니다. 모델에서 첫 레이어일 때만 정의하면 됩니다.
                (행, 열, 채널 수)로 정의합니다. 흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3으로 설정합니다.
                (N, 28,28,1) ->  batch_shape + (rows, cols, channels) 
- activation : 활성화 함수 설정합니다.
‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
‘relu’ : rectifier 함수, 은익층에 주로 쓰입니다.
‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.
‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.

'''
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D,Dense

# model = Sequential()
# model.add(Conv2D(filters = 10, kernel_size = (2,2), input_shape=(10,10,1)))# 실제 데이터의 크기는 (N,10,10,1)
# model.add(Dense(1))
# model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 9, 9, 10)          50
# _________________________________________________________________
# dense (Dense)                (None, 9, 9, 1)           11
# =================================================================
# Total params: 61
# Trainable params: 61
# Non-trainable params: 0

#################################################################################

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D,Dense,Flatten

# model = Sequential()
# model.add(Conv2D(filters = 10, kernel_size = (2,2), input_shape=(10,10,1)))# 실제 데이터의 크기는 (N,10,10,1)
# model.add(Flatten()) # -> 2차원으로 변경
# model.add(Dense(1))
# model.summary()

# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 9, 9, 10)          50
# _________________________________________________________________
# flatten (Flatten)            (None, 810)               0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 811
# =================================================================
# Total params: 861
# Trainable params: 861
# Non-trainable params: 0
#####################################################################################################
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten

model = Sequential()
model.add(Conv2D(filters = 10, kernel_size = (2,2), input_shape=(10,10,1)))
model.add(Conv2D(9,(2,2)))
model.add(Conv2D(9,(2,3))) # 크기 달라도 가능
model.add(Conv2D(8,2)) # 2라고만 써도 (2,2)로 인식
model.add(Flatten()) 
model.add(Dense(1))
model.summary()

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 8, 8, 9)           369
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 6, 9)           495
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 5, 8)           296
_________________________________________________________________
flatten (Flatten)            (None, 240)               0
_________________________________________________________________
dense (Dense)                (None, 1)                 241
=================================================================
Total params: 1,451
Trainable params: 1,451
Non-trainable params: 0
Non-trainable params: 0

# 2번 이상하면 특성 수치가 높아져서 성능 좋아짐. (통상적) 1번하는게 더 좋을 수도 있음. 판단은 내가. 
model.add(Conv2D(filters = 10, kernel_size = (2,2), input_shape=(10,10,1))) 의 Param = 50 / (N, 10,10,1) ->  (batch_shape ,rows, cols, channels) 
number_parameters = out_channels * (in_channels * kernel_h * kernel_w + 1)
                  = 10 *(1*2*2 +1) = 50
-> 다음 레이어로 전달하는 연산 횟수 
'''
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten

model = Sequential()
model.add(Conv2D(filters = 10, kernel_size = (2,2),strides=2,
                 padding='same',input_shape=(10,10,1))) 
model.add(Conv2D(9,(2,2)))
model.add(Flatten()) 
model.add(Dense(1))
model.summary()

# padding = ‘same’ : 출력 이미지 사이즈가 입력 이미지 사이즈와 동일 -> kernel size보다 강력
# padding 기본값 = valid
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 10, 10, 10)        50
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 9, 9, 9)           369
# _________________________________________________________________
# flatten (Flatten)            (None, 729)               0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 730
# =================================================================
# Total params: 1,149
# Trainable params: 1,149

#strides = 1 (1칸씩 간다), 2= (2칸씩 간다) / 기본값 = 1 
# - 특성이 중첩되는게 좋지만,, 다음 레이어에서는 필요없을 수도 있음. 데이터마다 다르기 때문에 알아서 판단 

'''
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D

model = Sequential()
model.add(Conv2D(filters = 10, kernel_size = (2,2),strides=1,
                 padding='same',input_shape=(10,10,1))) 
model.add(MaxPooling2D())
model.add(Conv2D(9,(2,2)))
model.add(Flatten()) 
model.add(Dense(1))
model.summary()

_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 10, 10, 10)        50
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 5, 5, 10)          0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 4, 4, 9)           369
_________________________________________________________________
flatten (Flatten)            (None, 144)               0
_________________________________________________________________
dense (Dense)                (None, 1)                 145
=================================================================
Total params: 564
Trainable params: 564
Non-trainable params: 0
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D

model = Sequential()
model.add(Conv2D(filters = 10, kernel_size = (2,2),strides=1,
                 padding='same',input_shape=(10,10,1))) 
model.add(MaxPooling2D(pool_size=(2,3)))                 
model.add(Conv2D(9,(2,2),padding = 'valid'))
model.add(Flatten()) 
model.add(Dense(1))
model.summary()

# MaxPooling2D : 전체 특성 중 제일 특성 강한 것만 빼고 나머지는 버림, 최초 conv 이후 어느때나 다 가능. 최초 conv 이전에는 쓸 수 없다. 
# pool_size = 2 = (2,2)
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 10, 10, 10)        50
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 5, 5, 10)          0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 4, 4, 9)           369
# _________________________________________________________________
# flatten (Flatten)            (None, 144)               0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 145
# =================================================================
# Total params: 564
# Trainable params: 564
# Non-trainable params: 0

# pool_size=3

# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 10, 10, 10)        50
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 3, 3, 10)          0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 2, 2, 9)           369
# _________________________________________________________________
# flatten (Flatten)            (None, 36)                0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 37
# =================================================================
# Total params: 456
# Trainable params: 456
# Non-trainable params: 0

# pool_size=(2,3)
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 10, 10, 10)        50
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 5, 3, 10)          0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 4, 2, 9)           369
# _________________________________________________________________
# flatten (Flatten)            (None, 72)                0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 73
# =================================================================
# Total params: 492
# Trainable params: 492
# Non-trainable params: 0