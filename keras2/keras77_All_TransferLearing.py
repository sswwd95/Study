from tensorflow.keras.applications import VGG16,VGG19
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import ResNet101,ResNet101V2,ResNet152,ResNet152V2,ResNet50,ResNet50V2
from tensorflow.keras.applications import InceptionV3,InceptionResNetV2
from tensorflow.keras.applications import MobileNet,MobileNetV2
from tensorflow.keras.applications import DenseNet121,DenseNet169,DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0,EfficientNetB1

model = VGG16()

model.trainable = False

model.summary()
print(len(model.weights)) #32
print(len(model.trainable_weights)) #32

'''

=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
'''

######################################

model = VGG19()

model.trainable = False

model.summary()
print(len(model.weights)) # 38
print(len(model.trainable_weights))

'''

=================================================================
Total params: 143,667,240
Trainable params: 0
Non-trainable params: 143,667,240
_________________________________________________________________
'''

######################################

model = Xception()

model.trainable = False

model.summary()
print(len(model.weights)) #236
print(len(model.trainable_weights))

'''

==================================================================================================
Total params: 22,910,480
Trainable params: 0
Non-trainable params: 22,910,480
__________________________________________________________________________________________________
'''

######################################

model = ResNet101()

model.trainable = False

model.summary()
print(len(model.weights)) #626
print(len(model.trainable_weights))
'''
==================================================================================================
Total params: 44,707,176
Trainable params: 0
Non-trainable params: 44,707,176
__________________________________________________________________________________________________
'''

######################################

model = ResNet101V2()

model.trainable = False

model.summary()
print(len(model.weights)) #544
print(len(model.trainable_weights))

'''

==================================================================================================
Total params: 60,419,944
Trainable params: 0
Non-trainable params: 60,419,944
__________________________________________________________________________________________________
'''

######################################

model = ResNet152V2()

model.trainable = False

model.summary()
print(len(model.weights))#816
print(len(model.trainable_weights))
'''
==================================================================================================
Total params: 60,380,648
Trainable params: 0
Non-trainable params: 60,380,648
__________________________________________________________________________________________________
'''


######################################

model = ResNet50()

model.trainable = False

model.summary()
print(len(model.weights))#320
print(len(model.trainable_weights))
'''
==================================================================================================
Total params: 25,636,712
Trainable params: 0
Non-trainable params: 25,636,712
__________________________________________________________________________________________________
'''

######################################

model = ResNet50V2()

model.trainable = False

model.summary()
print(len(model.weights)) #272
print(len(model.trainable_weights))

'''
==================================================================================================
Total params: 25,613,800
Trainable params: 0
Non-trainable params: 25,613,800
__________________________________________________________________________________________________
'''

######################################

model = InceptionV3()

model.trainable = False

model.summary()
print(len(model.weights))#378
print(len(model.trainable_weights))

'''
==================================================================================================
Total params: 23,851,784
Trainable params: 0
Non-trainable params: 23,851,784
__________________________________________________________________________________________________
'''


######################################

model = InceptionResNetV2()

model.trainable = False

model.summary()
print(len(model.weights))#898
print(len(model.trainable_weights))

'''
==================================================================================================
Total params: 55,873,736
Trainable params: 0
Non-trainable params: 55,873,736
__________________________________________________________________________________________________
'''

######################################

model = MobileNet()

model.trainable = False

model.summary()
print(len(model.weights)) #137
print(len(model.trainable_weights))
'''
=================================================================
Total params: 4,253,864
Trainable params: 0
Non-trainable params: 4,253,864
_________________________________________________________________
'''

######################################

model = MobileNetV2()

model.trainable = False

model.summary()
print(len(model.weights)) #262
print(len(model.trainable_weights))

'''
==================================================================================================
Total params: 3,538,984
Trainable params: 0
Non-trainable params: 3,538,984
__________________________________________________________________________________________________
'''

######################################

model = DenseNet121()

model.trainable = False

model.summary()
print(len(model.weights)) #606
print(len(model.trainable_weights))
'''
==================================================================================================
Total params: 8,062,504
Trainable params: 0
Non-trainable params: 8,062,504
__________________________________________________________________________________________________
'''

######################################

model = DenseNet169()

model.trainable = False

model.summary()
print(len(model.weights))#846
print(len(model.trainable_weights))
'''
==================================================================================================
Total params: 14,307,880
Trainable params: 0
Non-trainable params: 14,307,880
__________________________________________________________________________________________________
'''


######################################

model = DenseNet201()

model.trainable = False

model.summary()
print(len(model.weights))#1006
print(len(model.trainable_weights))
'''
==================================================================================================
Total params: 20,242,984
Trainable params: 0
Non-trainable params: 20,242,984
__________________________________________________________________________________________________
'''

######################################

model = NASNetLarge()

model.trainable = False

model.summary()
print(len(model.weights))#1546
print(len(model.trainable_weights))
'''
==================================================================================================
Total params: 88,949,818
Trainable params: 0
Non-trainable params: 88,949,818
__________________________________________________________________________________________________
'''

######################################

model = NASNetMobile()

model.trainable = False

model.summary()
print(len(model.weights)) #1126
print(len(model.trainable_weights))
'''
==================================================================================================
Total params: 5,326,716
Trainable params: 0
Non-trainable params: 5,326,716
__________________________________________________________________________________________________
'''

######################################

model = EfficientNetB0()

model.trainable = False

model.summary()
print(len(model.weights))#314
print(len(model.trainable_weights))
'''
==================================================================================================
Total params: 5,330,571
Trainable params: 0
Non-trainable params: 5,330,571
__________________________________________________________________________________________________
'''

######################################

model = EfficientNetB1()

model.trainable = False

model.summary()
print(len(model.weights))#442
print(len(model.trainable_weights))
'''
Total params: 7,856,239
Trainable params: 0
Non-trainable params: 7,856,239
__________________________________________________________________________________________________
'''
