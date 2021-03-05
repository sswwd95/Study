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
# Computer vision with CNNs
#
# Create and train a classifier for horses or humans using the provided data.
# Make sure your final layer is a 1 neuron, activated by sigmoid as shown.
#
# The test will use images that are 300x300 with 3 bytes color depth so be sure to
# design your neural network accordingly
import numpy as np
import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def solution_model():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    urllib.request.urlretrieve(_TRAIN_URL, './/tf_certificate//Category3/horse-or-human.zip')
    local_zip = './/tf_certificate//Category3/horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('.//tf_certificate//Category3/tmp/horse-or-human/')
    zip_ref.close()
    urllib.request.urlretrieve(_TEST_URL, './/tf_certificate//Category3/testdata.zip')
    local_zip = './/tf_certificate//Category3/testdata.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('.//tf_certificate//Category3/tmp2/testdata/')
    zip_ref.close()

    train_datagen = ImageDataGenerator(
        height_shift_range= 0.1,
        width_shift_range= 0.1,
        rescale= 1/255.)
        #Your code here. Should at least have a rescale. Other parameters can help with overfitting.)

    validation_datagen = ImageDataGenerator(
        rescale = 1/255.)#Your Code here)

    batch_size=16

    train_generator = train_datagen.flow_from_directory( 
        './/tf_certificate//Category3//tmp/horse-or-human/',
        target_size = (300,300),
        class_mode = 'binary',
        batch_size= batch_size)
        #Your Code Here)

    validation_generator = validation_datagen.flow_from_directory(
        './/tf_certificate//Category3//tmp2/testdata/',
        target_size = (300,300),
        class_mode = 'binary',
        batch_size= batch_size)
        #Your Code Here)


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'valid', input_shape = (300, 300, 3)),
        tf.keras.layers.MaxPooling2D(3,3),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', padding = 'valid'),
        tf.keras.layers.MaxPooling2D(3,3),
        tf.keras.layers.Conv2D(128, (5,5), activation = 'relu', padding = 'valid'),
        tf.keras.layers.MaxPooling2D(5,5),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dense(16, activation = 'relu'),
        # Note the input shape specified on your first layer must be (300,300,3)
        # Your Code here

        # This is the last layer. You should not change this code.
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    es = EarlyStopping(patience = 6)
    lr = ReduceLROnPlateau(factor = 0.25, verbose = 1, patience = 3)

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics= ['acc'])

    model.fit(train_generator, epochs = 1000, steps_per_epoch= np.ceil(1027/batch_size), validation_data= validation_generator,
    validation_steps = np.ceil(256/batch_size), callbacks = [es, lr])

    print(model.evaluate(validation_generator, steps = np.ceil(256/batch_size)))


    return model

    # NOTE: If training is taking a very long time, you should consider setting the batch size
    # appropriately on the generator, and the steps per epoch in the model.fit() function.

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save(".//tf_certificate//Category3/hosre-or-human.h5")

    # [1.4198849201202393, 0.82421875]