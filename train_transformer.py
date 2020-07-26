import pickle
import argparse
import numpy as np
import pandas as po

import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformer import TokenAndPositionEmbedding, TransformerBlock

def fit_tokenizer(args, df):
	df['Text'] = df['questionText'] + ' ' + df['answerText'] 

	texts = []
	texts += df['Text'].to_list()

	tokenizer = Tokenizer(num_words=args.vocab_size)
	tokenizer.fit_on_texts(texts)

	return tokenizer

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

def build_model(args):
	embedding_layer = TokenAndPositionEmbedding(args.max_seq_len, args.vocab_size, args.emb_dim)
	transformer_block = TransformerBlock(args.emb_dim, args.num_heads, args.ff_dim)

	inputs = layers.Input(shape=(args.max_seq_len,))
	x = embedding_layer(inputs)
	x = transformer_block(x)
	x = layers.GlobalAveragePooling1D()(x)
	x = layers.Dropout(0.1)(x)
	x = layers.Dense(20, activation="relu")(x)
	x = layers.Dropout(0.1)(x)
	outputs = layers.Dense(2, activation="softmax")(x)

	model = Model(inputs=inputs, outputs=outputs)
	optimizer = tf.keras.optimizers.Adam(lr=0.001)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	print(model.summary())

	return model

def main(args):
	df = po.read_csv(args.data_path).sample(frac=1)

	tokenizer = fit_tokenizer(args, df)

	(X_train, y_train), (X_val, y_val), (X_test, y_test) = get_train_val_test(args, df, tokenizer)

	model = build_model(args)

	earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
	
	model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[earlystop], shuffle=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser() 
	
	parser.add_argument('-data_path', type=str, default='data/automative_question_answers.csv')

	parser.add_argument('-vocab_size', type=int, default=1000000)
	parser.add_argument('-max_seq_len', type=int, default=50) #for default size -> average + 3 stds of lengths is 47

	parser.add_argument('-train_percent', type=float, default=0.7)
	parser.add_argument('-val_percent', type=float, default=0.9)

	parser.add_argument('-emb_dim', type=int, default=100) # Embedding size for each token
	parser.add_argument('-num_heads', type=int, default=2) # Number of attention heads
	parser.add_argument('-ff_dim', type=int, default=32) # Hidden layer size in feed forward network inside transformer

	args = parser.parse_args()

	main(args)


'''

# See paper for why positional embeddings are calculated like this
def get_angles(pos, i, d_model):
	angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
	return pos * angle_rates

def positional_encoding(position, d_model):
	angle_rads = get_angles(np.arange(position)[:, np.newaxis], 
							np.arange(d_model)[np.newaxis, :],
							d_model)

	# apply sin to even indices in the array; 2i
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

	# apply cos to odd indices in the array; 2i+1
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

	# add one new axis at the start, preserving other axes as is 
	pos_encoding = angle_rads[np.newaxis, ...]

	# change dtype
	return tf.cast(pos_encoding, dtype=tf.float32) 

def create_padding_mask(seq):
	seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

	# add extra dimensions to add the padding
	# to the attention logits.
	return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def main(args):
	df = po.read_csv(args.data_path).sample(frac=1)

	tokenizer = fit_tokenizer(args, df)

	sent = tokenizer.texts_to_sequences([df['Text'].iloc[0]])
	padded_sent = pad_sequences(sent, maxlen=args.max_seq_len)
	mask = create_padding_mask(padded_sent)

	print(sent)
	print(padded_sent)
	print(mask)
	
	#print(positional_encoding(20, 300).shape)
'''