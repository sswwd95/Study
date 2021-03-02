from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

model.trainable = False
model.summary()

print(len(model.weights)) #26
print(len(model.trainable_weights))

'''
=================================================================
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688
_________________________________________________________________
'''
#############################################################################################

# 만약 vgg16을 기본값으로 한다면?

from tensorflow.keras.applications import VGG16

model = VGG16()
# model = VGG16(weights='imagenet', include_top=True, input_shape=(224,224,3)) 와 동일


model.trainable = False
model.summary()

print(len(model.weights)) #32
print(len(model.trainable_weights))
'''
=================================================================
Total params: 138,357,544
Trainable params: 0
Non-trainable params: 138,357,544
_________________________________________________________________
'''