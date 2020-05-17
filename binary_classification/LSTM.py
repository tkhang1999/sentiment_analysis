from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from keras.optimizers import Adam

import pandas as pd
import numpy as np 

# Data pre-processing
data = pd.read_csv("../yelp.csv")
sentences = data['text'].values
y = data['label'].values

# Prepare train and test data
x_train,x_test,y_train,y_test = train_test_split(sentences,y,test_size=0.2,random_state=1000)

num_words=5000

embedding_size=32
optimizer = Adam(lr=1e-3)

tokenizer = Tokenizer(num_words = num_words)
tokenizer.fit_on_texts(sentences)
vocab_size=len(tokenizer.word_index)+1
print(vocab_size)

x_train_tokens = tokenizer.texts_to_sequences(x_train)
x_test_tokens = tokenizer.texts_to_sequences(x_test)
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

max_tokens = int(np.mean(num_tokens)+2*np.std(num_tokens))

x_train = pad_sequences(x_train_tokens,padding='pre',maxlen = max_tokens)
x_test = pad_sequences(x_test_tokens,padding='pre',maxlen=max_tokens)

print(y_train.shape)
print(x_train.shape)

# Create the model
model = Sequential()
model.add(Embedding(input_dim = vocab_size,output_dim=embedding_size,input_length=max_tokens))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
print(model.summary())

# Train and Test
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3,batch_size=128)
print("finish training")
result = model.evaluate(x_test,y_test)
print(result)