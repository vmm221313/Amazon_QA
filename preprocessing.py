import json
import pickle
import argparse
import numpy as np
import pandas as po
from tqdm import tqdm

def read_glove_file():
	with open('data/glove.6B.300d.txt', 'r') as f:
		l = f.read()

	word2emb = {}
	words_embs = l.split('\n')

	for i in tqdm(range(len(words_embs))):
		word = words_embs[i].split(' ')[0]
		emb = np.array(words_embs[i].split(' ')[1:]).astype(float)
		word2emb[word] = emb

	with open('data/word2emb.pkl', 'wb') as f:
		pickle.dump(word2emb, f)

def open_word2emb():
	with open('data/word2emb.pkl', 'rb') as f:
		word2emb = pickle.load(f)

def process_data():
	data = po.read_csv('data/automative_question_answers.csv')
	data = data.sample(frac=1)

	train_data = data[:int(0.8*len(data))]

	texts = []
	texts += data['questionText'].to_list()
	texts += data['answerText'].to_list()

	tokenizer = Tokenizer(num_words=vocab_size)
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)

	print(sequences[0])

	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))

	data = pad_sequences(sequences, maxlen=max_seq_len)

	print(data[0])

	print(len(sequences))
	print(len(data))

	#labels = to_categorical(np.asarray(labels))
	print('Shape of data tensor:', data.shape)
	#print('Shape of label tensor:', labels.shape)

def read_meta():
	with open('data/meta_Automotive.json', 'r') as f:
		meta = json.load(f)

	print(meta.keys())

 
if __name__ == '__main__':
	parser = argparse.ArgumentParser() 
	parser.add_argument('-function', type=str)

	args = parser.parse_args()

	expression = args.function+'()'
	eval(expression)
