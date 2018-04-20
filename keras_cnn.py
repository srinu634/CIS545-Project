import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout, LSTM, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import TensorBoard

#reference for code - http://www.developintelligence.com/blog/2017/06/practical-neural-networks-keras-classifying-yelp-reviews/
#reference for code - https://medium.com/@thoszymkowiak/how-to-implement-sentiment-analysis-using-word-embedding-and-convolutional-neural-networks-on-keras-163197aef623


# Using keras to load the dataset with the top_words
top_words = 10000

# Pad the sequence to the same length
max_review_length = 1600
X = []
y = []
with open('flyfrontier_tweets_with_sentiment.txt', 'r', encoding='utf8', errors='replace') as f:
	for line in f:
		if not line:
			continue
		line =  line.strip()
		X.append(str(line.split('\t')[0]))
		y.append(int(line.split('\t')[1]))

X = np.asarray(X)
y = np.asarray(y)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
data = sequence.pad_sequences(sequences, maxlen=max_review_length)

X_train, X_test, y_train, y_test = train_test_split(data, y, train_size=0.8)

# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# # Using embedding from Keras
embedding_vecor_length = 300
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Dropout(0.2))
model.add(Convolution1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128))
model.add(Dense(3,activation='softmax'))

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3, batch_size=64)

# Evaluation on the test set
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
