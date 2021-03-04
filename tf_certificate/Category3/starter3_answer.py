# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Computer Vision with CNNs
#
# Build a classifier for Rock-Paper-Scissors based on the rock_paper_scissors
# TensorFlow dataset.
#
# IMPORTANT: Your final layer should be as shown. Do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail. 
#
# NOTE THAT THIS IS UNLABELLED DATA. 
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.

# (150,150,3)
import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import numpy as np



# def solution_model():
#     url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
#     urllib.request.urlretrieve(url, 'C:\\Study\\tf_certificate\\Category3\\rps.zip')
#     local_zip = 'C:\\Study\\tf_certificate\\Category3\\rps.zip'
#     zip_ref = zipfile.ZipFile(local_zip, 'r')
#     # zip_ref.extractall('tmp/')
#     zip_ref.extractall('C:\\Study\\tf_certificate\\Category3\\tmp\\')
#     zip_ref.close()

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, './/tf_certificate//Category3//rps.zip')
    local_zip = './/tf_certificate//Category3//rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('.//tf_certificate//Category3//')
    zip_ref.close()


    TRAINING_DIR = ".//tf_certificate//Category3/rps/"
    training_datagen = ImageDataGenerator(rescale=1.0/255,
                                         rotation_range=10,
                                         width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         shear_range=0.1,
                                         zoom_range=0.1,
                                         horizontal_flip=True,
                                         fill_mode='nearest',
                                         validation_split=0.2)
    
    train_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                           batch_size=16,
                                                           class_mode='categorical',
                                                           target_size=(150,150),
                                                           subset='training')

    val_generator = training_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=16,
                                                    class_mode='categorical',
                                                    target_size=(150,150),
                                                    subset='validation')

    
    
    model = tf.keras.models.Sequential([
    # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(256, 3 , activation = 'relu', padding = 'same', input_shape = (150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.summary()
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    rl = ReduceLROnPlateau(monitor='val_loss', patience= 10, factor=0.5)
    es = EarlyStopping(monitor = 'val_loss', patience=20, mode='auto')

    from tensorflow.keras.optimizers import RMSprop,Adam

    model.compile(loss = 'categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])
    model.fit_generator(train_generator, epochs=500, steps_per_epoch= np.ceil(2016/16), validation_steps= np.ceil(504/16), validation_data=val_generator, callbacks=[es, rl])
    # np.ceil(a) : 각 원소 값보다 크거나 같은 가장 작은 정수 값 (천장 값)으로 올림

    loss, acc = model.evaluate(val_generator, batch_size=16)
    print('loss, acc : ', loss, acc)

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save(".//tf_certificate//Category3/mymodel.h5")

# loss, acc :  0.19466343522071838 0.966269850730896