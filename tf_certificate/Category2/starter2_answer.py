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
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # YOUR CODE HERE
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train/255.
    x_test = x_test/255.
    

    from tensorflow.keras.utils import to_categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D,Flatten, MaxPooling1D, Dropout

    model = Sequential()
    model.add(Conv1D(256,3, activation='relu', input_shape=(28,28)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.3))
    model.add(Conv1D(128,3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='acc', patience=20, mode='auto')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
    model.fit(x_train, y_train, batch_size=16, epochs=500, validation_split=0.2, callbacks=[es])

    loss, acc = model.evaluate(x_test, y_test, batch_size=16)
    print('loss, acc : ', loss, acc)


    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save(".//tf_certificate//Category2/mymodel.h5")

# loss, acc :  0.6094269752502441 0.8906000256538391
