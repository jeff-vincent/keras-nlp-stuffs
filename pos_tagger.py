import nltk
nltk.download('treebank')
import numpy as np
from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import Embedding, Activation
from keras.optimizers import Adam
from keras import backend as K
 
tagged_sentences = nltk.corpus.treebank.tagged_sents()

# separate POS tags from sentences
# in nltk corpus `tagged_sents()`
sentences, sentence_tags = [], [] 
for tagged_sentence in tagged_sentences:
    sentence, tags = zip(*tagged_sentence)
    sentences.append(np.array(sentence))
    sentence_tags.append(np.array(tags))


# break out training/testing data
#
# NOTE: `test_size` specifies the
# size of the testing dataset. 
# By default, it will be set to 0.25 
# of the size of training set
(train_sentences, 
 test_sentences, 
 train_tags, 
 test_tags) = train_test_split(sentences, sentence_tags, test_size=0.2)

words, tags = set([]), set([])
# add words to set
for s in train_sentences:
    for w in s:
        words.add(w.lower())
	
# add tags to set
for ts in train_tags:
    for t in ts:
        tags.add(t)
	
# armed with a set => list of all words in the training corpus
# generate a dict of index & associated word
word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0  # The special value used for padding
word2index['-OOV-'] = 1  # The special value used for OOVs
 
# dict of tags
tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0  # The special value used to padding

# make lists
train_sentences_X, test_sentences_X = [], []
train_tags_y, test_tags_y = [], []
 
# iterate over sententences
for s in train_sentences:
	
    s_int = [] # create s_int list
    for w in s:
        try:
            s_int.append(word2index[w.lower()]) # append list of lists -- each containing a single dict
        except KeyError:
            s_int.append(word2index['-OOV-'])
# append list of sdict
    train_sentences_X.append(s_int)
 
for s in test_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])
 
    test_sentences_X.append(s_int)
 
for s in train_tags:
    train_tags_y.append([tag2index[t] for t in s])
 
for s in test_tags:
    test_tags_y.append([tag2index[t] for t in s])

# identify longest sentence
MAX_LENGTH = len(max(train_sentences_X, key=len))
# pad to equalize varying lengths
train_sentences_X = pad_sequences(train_sentences_X, 
maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, 
maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, 
maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, 
maxlen=MAX_LENGTH, padding='post')

# define model
model = Sequential()

# define layer 
# NOTE: `input_shape` = This is a shape tuple 
# (a tuple of integers or None entries, where 
# None indicates that any positive integer may be expected).
# In input_shape, the batch dimension is not included.
model.add(InputLayer(input_shape=(MAX_LENGTH, )))


# define layer
# NOTE: Turns positive integers (indexes)
# into dense vectors of fixed size. 
# eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
# NOTE: This layer can only be used as the first layer in a model.
This layer can only be used as the first layer in a model.
model.add(Embedding(len(word2index), 128))

# long short term memory
# 
model.add(Bidirectional(LSTM(256, return_sequences=True)))
# dense layer
# Hidden Lots of _densely_ interconnected nodes... 
# NOTE: https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L116
model.add(TimeDistributed(Dense(len(tag2index))))
# softmax activation function
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])
 
model.summary()

# conceptual break ----------------------------------------------------------

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))

model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), 
batch_size=128, epochs=40, validation_split=0.2)

scores = model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")

test_samples = [
    "running is very important for me .".split(),
    "I was running every day for a month .".split()
]

test_samples_X = []
for s in test_samples:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])
    test_samples_X.append(s_int)
 
test_samples_X = pad_sequences(test_samples_X, maxlen=MAX_LENGTH, padding='post')

predictions = model.predict(test_samples_X)
print(predictions, predictions.shape)

def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences

def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy

model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word2index), 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy', ignore_class_accuracy(0)])
 
model.summary()

model.fit(train_sentences_X, to_categorical(train_tags_y, 
len(tag2index)), batch_size=128, epochs=40, validation_split=0.2)

predictions = model.predict(test_samples_X)
print(logits_to_tokens(predictions, {i: t for t, i in tag2index.items()}))
