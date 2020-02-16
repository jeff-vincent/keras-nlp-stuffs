from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.utils import plot_model

import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
import re

import matplotlib.pyplot as plt

# parse csv
toxic_comments = pd.read_csv("toxic_comments.csv")
# remove empty strings
filter = toxic_comments["comment_text"] != ""
toxic_comments = toxic_comments[filter]
# remove missing elements
toxic_comments = toxic_comments.dropna()

# Set labels on Pandas DataFrame
toxic_comments_labels = toxic_comments[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
toxic_comments_labels.head()

def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

# Prepare X data
X = []
sentences = list(toxic_comments["comment_text"])
for sen in sentences:
    X.append(preprocess_text(sen))
    
# define y
y = toxic_comments_labels.values


# train_test_split() is used to break out training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# SOURCE: https://stackoverflow.com/questions/51956000/what-does-keras-tokenizer-method-exactly-do
# *fit_on_texts* Updates internal vocabulary based on a list of texts. 
# This method creates the vocabulary index based on word frequency. 
# So if you give it something like, "The cat sat on the mat." 
# It will create a dictionary s.t. word_index["the"] = 1; word_index["cat"] = 2 it is word -> 
# index dictionary so every word gets a unique integer value. 0 is reserved for padding. 
# So lower integer means more frequent word 
# (often the first few are stop words because they appear a lot).
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# *texts_to_sequences* Transforms each text in texts to a sequence of integers. 
# So it basically takes each word in the text and replaces it with its 
# corresponding integer value from the word_index dictionary. 
# Nothing more, nothing less, certainly no magic involved.
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# total count plus one of inclusive lexicon
vocab_size = len(tokenizer.word_index) + 1

# normalize length of text sequences
maxlen = 200
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# this dict will house our word vectors
embeddings_dictionary = dict()

# import pre-trained vectors
glove_file = open('/content/glove.6B.100d.txt', encoding="utf8")

# collect all pre-trained vectors that appear in test data
# and store them in the embeddings_dictionary
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

# build matrix of word vectors
embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# heavy lifting -- defining Layers & Model
deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(6, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

# Define loss function in `.compile()` call
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# simple diagram
print(model.summary())

# generate png description of Model
plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)

# run it
history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.2)

# test it
score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

# draw it
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
