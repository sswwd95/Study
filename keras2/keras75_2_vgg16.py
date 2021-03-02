from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

VGG16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

# model.trainable = False
# model.summary()
# print(len(model.weights)) #26 (13개의 레이어가 있다)
# print(len(model.trainable_weights)) # 0
# -----------------------------------------
# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688
# -----------------------------------------

# model.summary()
# print(len(model.weights)) #26 (13개의 레이어가 있다)
# print(len(model.trainable_weights)) # 26
# -----------------------------------------
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# -----------------------------------------

VGG16.trainable = False
VGG16.summary()
print(len(VGG16.weights)) #26 (13개의 레이어가 있다)
print(len(VGG16.trainable_weights)) # 0

model = Sequential()
model.add(VGG16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1)) #, activation = 'softmax'))
model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
vgg16 (Functional)           (None, 1, 1, 512)         14714688
_________________________________________________________________
flatten (Flatten)            (None, 512)               0
_________________________________________________________________
dense (Dense)                (None, 10)                5130
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 6
=================================================================
Total params: 14,719,879
Trainable params: 5,191
Non-trainable params: 14,714,688
_________________________________________________________________
'''

print("그냥 가중치의 수  : ",len(model.weights)) #32
print("동결한 후 훈련되는 가중치의 수 : ",len(model.trainable_weights)) # 6

