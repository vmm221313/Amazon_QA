import pickle
import argparse
import numpy as np
import pandas as po

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.layers import Embedding, Input, Dense, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, TimeDistributed, LSTM, Bidirectional, Reshape
from tensorflow.keras.layers import Embedding, Input, Dense, Conv2D, Reshape, MaxPooling2D, Flatten, Dropout

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from sklearn.metrics import f1_score

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def fit_tokenizer(args, df):
	df['Text'] = df['questionText'] + ' ' + df['answerText'] 

	texts = []
	texts += df['Text'].to_list()

	tokenizer = Tokenizer(num_words=args.vocab_size)
	tokenizer.fit_on_texts(texts)

	return tokenizer

def make_embedding_layer(args, tokenizer):
	word_index = tokenizer.word_index

	with open('data/word2emb.pkl', 'rb') as f:
		word2emb = pickle.load(f)

	embedding_matrix = np.zeros((len(word_index) + 1, 300)) # words not found in embedding index will be all-zeros.
	for word, i in word_index.items():
		embedding_vector = word2emb.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	GloVe = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=args.max_seq_len, trainable=True)

	return GloVe

def get_train_val_test(args, df, tokenizer):
	sequences = tokenizer.texts_to_sequences(df['Text'].to_list())
	data = pad_sequences(sequences, maxlen=args.max_seq_len)

	labels = df['Helpfulness_Target'].to_list()
	labels = to_categorical(np.asarray(labels))	

	X_train = data[:int(args.train_percent*len(data))]
	X_val = data[int(args.train_percent*len(data)):int(args.val_percent*len(data))]
	X_test = data[int(args.val_percent*len(data)):]

	y_train = labels[:int(args.train_percent*len(labels))]
	y_val = labels[int(args.train_percent*len(labels)):int(args.val_percent*len(labels))]
	y_test = labels[int(args.val_percent*len(labels)):]

	return (X_train, y_train), (X_val, y_val), (X_test, y_test) 

def build_model(args, GloVe):
	sequence_input = Input(shape=(args.max_seq_len,), dtype='int32')
	
	x = GloVe(sequence_input)
	x = Reshape((50, 300, 1))(x)
	x = Conv2D(256, 5, activation='tanh')(x)
	x = MaxPooling2D((46, 296))(x)
	x = Flatten()(x)
	x = Dropout(0.2)(x)
	preds = Dense(2, activation='softmax')(x)

	model = Model(sequence_input, preds)
	optimizer = tf.keras.optimizers.Adam(lr=0.001)

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	print(model.summary())

	return model

def main(args):
	df = po.read_csv(args.data_path).sample(frac=1)

	tokenizer = fit_tokenizer(args, df)
	GloVe = make_embedding_layer(args, tokenizer)

	(X_train, y_train), (X_val, y_val), (X_test, y_test) = get_train_val_test(args, df, tokenizer)
		
	#print(df['Helpfulness_Target'].value_counts())
	
	model = build_model(args, GloVe)

	earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
	
	print(np.bincount(np.argmax(y_train, axis=1)))
	print(np.bincount(np.argmax(y_val, axis=1)))
	print(np.bincount(np.argmax(y_test, axis=1)))

	#'''
	model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[earlystop], shuffle=True)

	y_pred = model.predict(X_test)
	#print(y_pred[:100])

	y_pred = np.argmax(y_pred, axis = 1)
	y_test = np.argmax(y_test, axis = 1)
	print(f1_score(y_test, y_pred))
	#'''

if __name__ == '__main__':
	parser = argparse.ArgumentParser() 
	
	parser.add_argument('-data_path', type=str, default='data/automative_question_answers.csv')

	parser.add_argument('-vocab_size', type=int, default=1000000)
	parser.add_argument('-max_seq_len', type=int, default=50) #for default size -> average + 3 stds of lengths is 47

	parser.add_argument('-train_percent', type=float, default=0.7)
	parser.add_argument('-val_percent', type=float, default=0.9)
	#parser.add_argument('-test_percent', type=float, default=0.1)

	args = parser.parse_args()

	main(args)

'''

print('##', len(tokenizer.word_index))

print(sequences[0])

word_index 
print('Found %s unique tokens.' % len(word_index))



print(data[0])

print(len(sequences))
print(len(data))

#labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)
'''