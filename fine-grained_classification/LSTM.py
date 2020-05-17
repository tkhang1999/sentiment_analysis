from sklearn.model_selection import train_test_split 
from keras.layers import Dense, Embedding, LSTM, Dropout, Input, Bidirectional, GlobalMaxPool1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Model

import pandas as pd
import numpy as np

# READ DATA

SEED = 42
df = pd.read_csv('../yelp_food_review.csv')
# Use 10,000 sample data for faster training
# The dataset has more than 700,000 entries
rev_1 = df[df['stars'] == 1].sample(n = 2000)
rev_2 = df[df['stars'] == 2].sample(n = 2000)
rev_3 = df[df['stars'] == 3].sample(n = 2000)
rev_4 = df[df['stars'] == 4].sample(n = 2000)
rev_5 = df[df['stars'] == 5].sample(n = 2000)

dataset = pd.concat([rev_1, rev_2, rev_3, rev_4, rev_5]).sample(frac=1).reset_index(drop=True)

dataset['stars'] = dataset['stars'] - 1
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['stars'], test_size=0.1, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/9, random_state=SEED)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

# DATA PRE-PROCESSING

EMBEDDING_FILE='../glove.6B/glove.6B.50d.txt'

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 20000
# Max number of words in each review.
MAX_SEQUENCE_LENGTH = 100
# This is fixed.
EMBEDDING_DIM = 50
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['text'])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

x_train_tokens = tokenizer.texts_to_sequences(X_train)
x_val_tokens = tokenizer.texts_to_sequences(X_val)
x_test_tokens = tokenizer.texts_to_sequences(X_test)
num_tokens = [len(tokens) for tokens in x_train_tokens + x_val_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

x_train = pad_sequences(x_train_tokens, maxlen = MAX_SEQUENCE_LENGTH)
x_val = pad_sequences(x_val_tokens, maxlen = MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(x_test_tokens, maxlen=MAX_SEQUENCE_LENGTH)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding="utf8"))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

# CREATE MODEL

inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(5, activation="softmax")(x)
lstm_model = Model(inputs=inp, outputs=x)
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(lstm_model.summary())

# TRAIN MODEL

epochs = 6
batch_size = 32

history = lstm_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=[EarlyStopping(monitor='val_loss',patience=1, min_delta=0.0001)])
result = lstm_model.evaluate(x_test,y_test)
print(result)