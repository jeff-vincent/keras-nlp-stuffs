from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import re
import pandas as pd
from sklearn.model_selection import train_test_split
 
def clean_review(text):
    # Strip HTML tags
    text = re.sub('<[^<]+?>', ' ', text)
 
    # Strip escaped quotes
    text = text.replace('\\"', '')
 
    # Strip quotes
    text = text.replace('"', '')
 
    return text


# load dataframe in pandas
df = pd.read_csv('labeledTrainData.tsv', sep='\t', quoting=3)

# clean text to be analyzed
df['cleaned_review'] = df['review'].apply(clean_review)

# sklearn's `train_test_split()` -- standard for breaking 
# labeled data into train/test groups
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2)
 
vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'), 
                             lowercase=True, min_df=3, max_df=0.9, max_features=5000)
X_train_onehot = vectorizer.fit_transform(X_train)


# Sequential model for nlp baybee!!
model = Sequential()

# Define Embedding layer
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    64,  # Embedding size
                    input_length=MAX_SEQ_LENGHT))

# Define LSTM (Long Short Term Memory) layer
model.add(LSTM(64))

# Define Dense Layer with sigmoid function as activation function
# NOTE:
# In order to map predicted values to probabilities, 
# we use the Sigmoid function. The function maps any real value 
# into another value between 0 and 1. In machine learning, 
# we use sigmoid to map predictions to probabilities.
# SOURCE: https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train the Model
model.fit(X_train_onehot[:-100], y_train[:-100], 
          epochs=2, batch_size=128, verbose=1, 
          validation_data=(X_train_onehot[-100:], y_train[-100:]))



 