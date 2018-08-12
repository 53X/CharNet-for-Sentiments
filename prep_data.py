import string
import os
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np


alphabet = list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + ['\n']
char_index = {}

for i, character in enumerate(alphabet, start=1):
	char_index[i] = character

def create_dataset(path='dbpedia_data'):

	texts = []
	labels = []

	msg = 'Parsing the ' + path.upper() + ' dataset'

	print(msg)

	for i, c in enumerate(os.listdir(path)):

		with open(os.path.join(path,c,'data.txt'),'r') as file:

			for lines in file:

				texts.append(lines.lower())
				labels.append(i)

	print('Gathered {} samples and {} labels'.format(len(texts), len(labels)))
	print('{} classes found'.format(i+1))

	return texts, labels			


def structure_data(path='agnews_data'):

	texts, labels = create_dataset(path)
	tok =Tokenizer(char_level=True, split='')
	tok.fit_on_texts(texts)
	tok.word_index = char_index
	sequences = tok.texts_to_sequences(texts)
	padding = pad_sequences(sequences, maxlen=1024, padding='post')
	padding = np.array(padding)
	labels = to_categorical(labels)

	print('Annotations done and Data is ready to be fed to the network')

	return padding, labels


	
if __name__ =='__main__':

	structure_data()


