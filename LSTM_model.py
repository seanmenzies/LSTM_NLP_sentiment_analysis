import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import read_data

# load data
fname = r'C:\Users\Sean-work\OneDrive\Coding\PycharmProjects\NLP_btcsentanalysis\reviews\Reviews.csv'
df = pd.read_csv(fname)

# add sentiment column and clean data
df = read_data.gen_sentiment(df)
df['Text'] = df['Text'].apply(read_data.remove_punctuation)
df['Summary'] = df['Summary'].apply(read_data.remove_punctuation)

# split data into testing and training
text_list = df['Text'].to_list()
sent_list = df['Sentiments'].to_list()
split_point = int(len(text_list) * 0.8)
training_data = text_list[0:split_point]
training_labels = sent_list[0:split_point]
testing_data = text_list[split_point:]
testing_labels = sent_list[split_point:]

# tokenize and pad
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(training_data)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_data)
training_padded = pad_sequences(training_sequences, maxlen=150)
testing_sequences = tokenizer.texts_to_sequences(testing_data)
testing_padded = pad_sequences(testing_sequences, maxlen=150)

# convert list to np arrays
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 128),  # vocab size & dimension
    tf.keras.layers.LSTM(64, input_shape=(420651, 150, 1), return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='tanh')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(training_padded, training_labels, epochs=1,
                    validation_data=(testing_padded, testing_labels), verbose=1)
