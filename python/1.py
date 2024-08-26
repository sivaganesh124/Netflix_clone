import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample dataset (texts and labels)
texts = ["I love this movie", "I hate this movie", "This movie is okay"]
labels = [1, 0, 1]  # 1 for positive, 0 for negative

# Text preprocessing
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=10)

# Model architecture
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=10))
model.add(LSTM(units=64))
model.add(Dense(1, activation='sigmoid'))

# Model compilation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
model.fit(data, np.array(labels), epochs=10, batch_size=2)

# Model evaluation
loss, accuracy = model.evaluate(data, np.array(labels))
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Make predictions
new_texts = ["I really love this movie", "This movie was terrible"]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_data = pad_sequences(new_sequences, maxlen=10)
predictions = model.predict(new_data)
print(predictions)
