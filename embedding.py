import string
import os

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


	
if __name__ =='__main__':

	create_dataset()


