import os
import pickle
import argparse
import numpy as np
import pandas as po

import tensorflow as tf
from tensorflow.keras.models import Model
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
	#df['Text'] = df['questionText'] + ' ' + df['answerText'] 

	texts = []
	texts += df['text'].to_list()
	texts += df['meta'].to_list()

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
	sequences 		= tokenizer.texts_to_sequences(df['text'].to_list())
	meta 			= tokenizer.texts_to_sequences(df['meta'].to_list())

	data 			= pad_sequences(sequences, maxlen=args.max_seq_len)
	meta 			= pad_sequences(meta, maxlen=args.max_seq_len)

	labels 			= df['target'].to_list()
	labels 			= to_categorical(np.asarray(labels))	

	stacked_data = []
	for i in range(len(data)):
		stacked_data.append(np.vstack((data[i], meta[i])))
		
	stacked_data = np.array(stacked_data)

	X_train 		= stacked_data[:int(args.train_percent*len(stacked_data))].reshape(-1, args.max_seq_len, 2)
	X_val 			= stacked_data[int(args.train_percent*len(stacked_data)):int(args.val_percent*len(stacked_data))].reshape(-1, args.max_seq_len, 2)
	X_test			= stacked_data[int(args.val_percent*len(stacked_data)):].reshape(-1, args.max_seq_len, 2)

	y_train 		= labels[:int(args.train_percent*len(labels))]
	y_val 			= labels[int(args.train_percent*len(labels)):int(args.val_percent*len(labels))]
	y_test 			= labels[int(args.val_percent*len(labels)):]

	return (X_train, y_train), (X_val,  y_val), (X_test, y_test)

def build_model(args, GloVe, use_meta):
	sequence_input = Input(shape=(args.max_seq_len, 2), dtype='int32')

	text_in = sequence_input[:, :, 0]
	x = GloVe(text_in)
	x = Reshape((350, 300, 1))(x)
	x = Conv2D(256, 5, activation='tanh')(x)
	x = MaxPooling2D((46, 296))(x)
	x = Flatten()(x)
	x = Dropout(0.2)(x)
	
	meta_in = sequence_input[:, :, 1]
	m = GloVe(meta_in)
	m = Reshape((350, 300, 1))(m)
	m = Conv2D(256, 5, activation='tanh')(m)
	m = MaxPooling2D((46, 296))(m)
	m = Flatten()(m)
	m = Dropout(0.2)(m)
	
	if use_meta:
		out = tf.keras.backend.concatenate((x, m), axis=1)
	
	else:
		out = x

	preds = Dense(2, activation='softmax')(out)

	model = Model(sequence_input, preds)
	
	optimizer = tf.keras.optimizers.Adam(lr=0.001)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	model.summary()

	return model

def preprocess_json(json_path, data_path):
	print('Processed data for dataset not found, preprocessing json file {}'.format(json_path))

	with open('data/data_v2/Auto_meta_qar.pkl', 'rb') as f:
		data = pickle.load(f)

	X = []
	y = []

	df = po.DataFrame()
	for key in tqdm(data, total=len(data)):
		row = data[key]

		meta = ' '.join([row['title'], ' '.join(row['category']), ' '.join(row['description'])])
		
		for i in range(len(row['questions_answers'])):
			df_row = {}
			df_row['meta'] = meta

			ques = row['questions_answers'][i][0]
			reviews = ' '.join(row['questions_answers'][i][2])
			df_row['text'] = ' '.join([ques, reviews])

			target = row['questions_answers'][i][1]
			if target == 'Y':
				df_row['target'] = 1 
			elif target == 'N':
				df_row['target'] = 0
			else:
				raise ValueError
		
			df = df.append(df_row, ignore_index=True)

	print('Saved Processed data at {}'.format(data_path))
	
	df.to_csv(data_path, index=False)

def get_data(dataset_name):
	if dataset_name == 'auto':
		data_path = 'data/Auto_meta_qar.csv'
		json_path = 'data/raw_json_data/Auto_meta_qar.pkl'
	elif dataset_name == 'electronics':
		data_path = 'data/electronics_meta_qar.csv'
		json_path = 'data/raw_json_data/electronics_meta_qar.pkl'
	elif dataset_name == 'home':
		data_path = 'data/home_meta_qar.csv'
		json_path = 'data/raw_json_data/home_meta_qar.pkl'

	if not os.path.exists(data_path):
		preprocess_data(json_path, data_path)

	print('Loading dataset {}'.format(data_path))
	df = po.read_csv(data_path).sample(frac=1)

	return df

def main(args):
	print('Running CNN model for {} dataset'.format(args.dataset_name))
	print('Using Metadata - {}'.format(args.use_meta))

	df = get_data(args.dataset_name)

	tokenizer = fit_tokenizer(args, df)
	GloVe = make_embedding_layer(args, tokenizer)

	(X_train, y_train), (X_val,  y_val), (X_test, y_test) = get_train_val_test(args, df, tokenizer)

	#'''
	model = build_model(args, GloVe, use_meta=args.use_meta)

	earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
	model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='cnn_{}_{}_best_model.hdf5'.format(args.dataset_name, args.use_meta), monitor='val_loss', save_best_only=True, save_weights_only=True)

	hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=4, callbacks=[earlystop, model_checkpoint], shuffle=True)

	os.makedirs('training_history', exist_ok=True)
	with open('training_history/cnn_{}_{}_best_model.hdf5'.format(args.dataset_name, args.use_meta), 'wb') as f:
		pickle.dump(hist.history, f)

	y_pred = model.predict(X_test)

	y_pred = np.argmax(y_pred, axis = 1)
	y_test = np.argmax(y_test, axis = 1)
	print(f1_score(y_test, y_pred))

	if not os.path.exists('results.csv'):
		results_df = po.DataFrame()
	else:
		results_df = po.read_csv('results.csv')

	row = {}
	row['model'] = 'cnn'
	row['dataset_name'] = args.dataset_name
	row['use_meta'] = args.use_meta
	row['final f1_score'] = f1_score(y_test, y_pred)
	results_df = results_df.append(row, ignore_index=True)
	
	results_df.to_csv('results.csv', index=False)

	#'''

if __name__ == '__main__':
	parser = argparse.ArgumentParser() 
		
	parser.add_argument('-dataset_name', type=str, default='auto', choices=['auto', 'electronics', 'home'])
	parser.add_argument('-use_meta', type=bool, default=False, choices=[True, False])

	#parser.add_argument('-data_path', type=str, default='data/electronics_meta_qar.csv')
	parser.add_argument('-vocab_size', type=int, default=1000000)
	parser.add_argument('-max_seq_len', type=int, default=350) #for default size -> average + 3 stds of lengths is 350

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