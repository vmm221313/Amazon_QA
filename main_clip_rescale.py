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
from tensorflow.keras.layers import Embedding, Input, Dense, Conv2D, Reshape, MaxPooling2D, Flatten, Dropout, Bidirectional, LSTM, MaxPooling1D, Multiply

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from sklearn.metrics import f1_score

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def fit_tokenizer(args, df):
	#df['Text'] = df['questionText'] + ' ' + df['answerText'] 

	texts = []
	texts += df['ques'].to_list()
	texts += df['meta'].to_list()

	for i in range(args.num_reviews):
		texts += df['review_{}'.format(i)].to_list()
		
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
	questions 		= tokenizer.texts_to_sequences(df['ques'].to_list())
	meta 			= tokenizer.texts_to_sequences(df['meta'].to_list())

	reviews 		= []
	for i in range(args.num_reviews):
		reviews.append(pad_sequences(tokenizer.texts_to_sequences(df['review_{}'.format(i)].to_list()), maxlen=args.max_seq_len).tolist())

	data 			= pad_sequences(questions, maxlen=args.max_seq_len).tolist()
	meta 			= pad_sequences(meta, maxlen=args.max_seq_len).tolist()

	labels 			= df['target'].to_list()
	labels 			= to_categorical(np.asarray(labels))	

	stacked_data = []
	for i in range(len(data)):
		data_to_stack = [data[i], meta[i]] + [rev[i] for rev in reviews]
		stacked_data.append(data_to_stack)
	stacked_data = np.array(stacked_data)

	X_train 		= stacked_data[:int(args.train_percent*len(stacked_data))].reshape(-1, args.max_seq_len, 2+args.num_reviews)
	X_val 			= stacked_data[int(args.train_percent*len(stacked_data)):int(args.val_percent*len(stacked_data))].reshape(-1, args.max_seq_len, 2+args.num_reviews)
	X_test			= stacked_data[int(args.val_percent*len(stacked_data)):].reshape(-1, args.max_seq_len, 2+args.num_reviews)

	y_train 		= labels[:int(args.train_percent*len(labels))]
	y_val 			= labels[int(args.train_percent*len(labels)):int(args.val_percent*len(labels))]
	y_test 			= labels[int(args.val_percent*len(labels)):]

	return (X_train, y_train), (X_val,  y_val), (X_test, y_test)

def build_model(args, GloVe):
	sequence_input 		= Input(shape=(args.max_seq_len, 2+args.num_reviews), dtype='int32')

	ques 				= sequence_input[:, :, 0] # question
	e_q 				= GloVe(ques)
	V_q 				= Bidirectional(LSTM(100, return_sequences=True))(e_q)
	x_q 				= MaxPooling1D(pool_size=args.max_seq_len)(V_q)
	x_q 				= tf.keras.backend.squeeze(x_q, axis=1) # contextualized_question_embedding
	
	meta_in 			= sequence_input[:, :, 1]
	m 					= GloVe(meta_in)
	m 					= Reshape((350, 300, 1))(m)
	m 					= Conv2D(256, 5, activation='tanh')(m)
	m 					= MaxPooling2D((46, 296))(m)
	m 					= Flatten()(m)
	m 					= Dense(200)(m)
	m 					= Dropout(0.2)(m)
	m 					= tf.keras.backend.expand_dims(m, axis=2)
	
	#'''
	reviews_x_c = []
	for i in range(args.num_reviews):
		reviews 		= sequence_input[:, :, 2+i] # reviews
		e_c 			= GloVe(reviews)
		V_c 			= Bidirectional(LSTM(100, return_sequences=True))(e_c)
		V_c__x_q 		= tf.keras.backend.batch_dot(V_c, x_q)
		
		beta 			= Dense(args.max_seq_len, activation='softmax')(V_c__x_q)
		values, indices = tf.math.top_k(beta, k=5, sorted=True)
		kth_highest 	= tf.reshape(values[:, -1], (-1, 1))
		m 				= tf.keras.backend.cast(tf.where(beta>=kth_highest, 1, 0), float)
		beta_dash 		= tf.keras.backend.l2_normalize(Multiply()([beta, m]), axis=-1)
		x_c 			= tf.keras.backend.batch_dot(beta_dash, V_c)
		x_c 			= tf.keras.backend.expand_dims(x_c, axis=-1)

		reviews_x_c.append(x_c)
	
	rev_attn 			= tf.keras.backend.concatenate(reviews_x_c, axis=2)
	m 					= tf.keras.backend.expand_dims(m, axis=2)
	x_q 				= tf.keras.backend.expand_dims(x_q, axis=2)

	#final_layer 		= tf.keras.backend.concatenate([x_q, m], axis=2)
	final_layer 		= tf.keras.backend.concatenate([x_q, rev_attn], axis=2)
	final_layer 		= tf.keras.backend.squeeze(Dense(1, activation='softmax')(final_layer), axis=-1)
	preds 				= Dense(2, activation='softmax')(final_layer)
	#'''

	model 				= Model(sequence_input, preds)
	optimizer 			= tf.keras.optimizers.Adam(lr=0.001)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	tf.keras.utils.plot_model(model, to_file='clip_rescale_attention.png', show_shapes=True)

	model.summary()

	return model

def main(args):
	df = po.read_csv(args.data_path).sample(frac=1)

	tokenizer = fit_tokenizer(args, df)
	GloVe = make_embedding_layer(args, tokenizer)

	(X_train, y_train), (X_val,  y_val), (X_test, y_test) = get_train_val_test(args, df, tokenizer)

	model = build_model(args, GloVe)
	
	#'''
	earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
	model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='clip_rescale_best_model.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=True)

	model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=4, callbacks=[earlystop, model_checkpoint], shuffle=True)

	y_pred = model.predict(X_test)
	#print(y_pred[:100])

	y_pred = np.argmax(y_pred, axis = 1)
	y_test = np.argmax(y_test, axis = 1)
	print(f1_score(y_test, y_pred))
	#'''

if __name__ == '__main__':
	parser = argparse.ArgumentParser() 
	
	parser.add_argument('-data_path', type=str, default='data/Auto_meta_qar_clip_rescale.csv')

	parser.add_argument('-vocab_size', type=int, default=1000000)
	parser.add_argument('-max_seq_len', type=int, default=50) #for default size -> average + 3 stds of lengths is 50

	parser.add_argument('-train_percent', type=float, default=0.7)
	parser.add_argument('-val_percent', type=float, default=0.9)
	#parser.add_argument('-test_percent', type=float, default=0.1)

	parser.add_argument('-num_reviews', type=int, default=10)

	args = parser.parse_args()

	main(args)




'''
	print(stacked_data.shape)

	print('ques', data[134])
	print(stacked_data[134][0])

	print('meta', meta[134])
	print(stacked_data[134][1])
	
	print('review_3', reviews[2][134])
	print(stacked_data[134][4])
'''


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