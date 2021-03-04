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
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, './/tf_certificate//Category4/sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open('.//tf_certificate//Category4/sarcasm.json') as file : 
        data = json.load(file)
    for item in data:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
    
    token = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    token.fit_on_texts(sentences)
    sentences = token.texts_to_sequences(sentences)
    sentences = pad_sequences(sentences, maxlen=max_length, padding=padding_type, truncating= trunc_type)

    print(len(sentences))
    x_train = np.array(sentences[0:training_size])
    x_test = np.array(sentences[training_size:])
    y_train = np.array(labels[0:training_size])
    y_test = np.array(labels[training_size:])

  
    model = tf.keras.Sequential([
    # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', patience=20)
    lr = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
    model.fit(x_train, y_train, epochs=500, validation_split=0.2, callbacks=[es,lr])

    result = model.evaluate(x_test, y_test)
    print('loss : ',result[0], 'acc : ',result[1])

 
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save(".//tf_certificate//Category4/mymodel.h5")

# [0.7219855785369873, 0.8045908212661743]
