import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 각 쉐입에 따라 모델이 달라진다
model = ak.ImageClassifier(
    # overwrite=True, # true, false 큰 효과 모르겠다
    max_trials=2 # 최대 시도 2번
)

model.fit(x_train, y_train, epochs=10) 

result = model.evaluate(x_test, y_test)

print(result)

# 실행 폴더 하단에 체크포인트 생긴다
# 위치 지정해라
'''
Trial 1 Complete [00h 01m 37s]
val_loss: 0.03961878642439842

Best val_loss So Far: 0.03961878642439842
Total elapsed time: 00h 01m 37s

Search: Running Trial #2

Hyperparameter    |Value             |Best Value So Far
image_block_1/b...|resnet            |vanilla
image_block_1/n...|True              |True
image_block_1/a...|True              |False
image_block_1/i...|True              |None
image_block_1/i...|True              |None
image_block_1/i...|0                 |None
image_block_1/i...|0                 |None
image_block_1/i...|0.1               |None
image_block_1/i...|0                 |None
image_block_1/r...|False             |None
image_block_1/r...|resnet50          |None
image_block_1/r...|True              |None
classification_...|global_avg        |flatten
classification_...|0                 |0.5
optimizer         |adam              |adam
learning_rate     |0.001             |0.001
'''
'''
Search: Running Trial #1

Hyperparameter    |Value             |Best Value So Far
image_block_1/b...|vanilla           |?
image_block_1/n...|True              |?
image_block_1/a...|False             |?
image_block_1/c...|3                 |?
image_block_1/c...|1                 |?
image_block_1/c...|2                 |?
image_block_1/c...|True              |?
image_block_1/c...|False             |?
image_block_1/c...|0.25              |?
image_block_1/c...|32                |?
image_block_1/c...|64                |?
classification_...|flatten           |?
classification_...|0.5               |?
optimizer         |adam              |?
learning_rate     |0.001             |?

Trial 1 Complete [00h 00m 53s]
val_loss: 0.038371145725250244

Best val_loss So Far: 0.038371145725250244
Total elapsed time: 00h 00m 53s

Search: Running Trial #2

Hyperparameter    |Value             |Best Value So Far
image_block_1/b...|resnet            |vanilla
image_block_1/n...|True              |True
image_block_1/a...|True              |False
image_block_1/i...|True              |None
image_block_1/i...|True              |None
image_block_1/i...|0                 |None
image_block_1/i...|0                 |None
image_block_1/i...|0.1               |None
image_block_1/i...|0                 |None
image_block_1/r...|False             |None
image_block_1/r...|resnet50          |None
image_block_1/r...|True              |None
classification_...|global_avg        |flatten
classification_...|0                 |0.5
optimizer         |adam              |adam
learning_rate     |0.001             |0.001
[0.03435850143432617, 0.989300012588501]

'''