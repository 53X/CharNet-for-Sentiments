import keras
from keras.layers import Conv1D, MaxPooling1D, Dense, Input, Embedding



def character_model(vocab_size=69, maxlen=1024):

	input_layer = Input(shape=(maxlen, ), name='sentence input')

	embedding = Embedding(input_dim = vocab_size+1, output_dim = 70, name='character embedding')(input_layer)

	conv_1 = Conv1D(filters=1024, kernel_size=7,  padding='valid',
					activation='relu', name='first conv')(embedding)

	pool_1 = MaxPooling1D(pool_size=3, padding='valid', name='first pool')(conv_1)

	conv_2 = Conv1D(filters=1024, kernel_size=7,  padding='valid',
					activation='relu', name='second conv')(pool_1)

	pool_2 = MaxPooling1D(pool_size=3, padding='valid', name='second pool')(conv_2)

	conv_3 = Conv1D(filters=1024, kernel_size=3,  padding='valid',
					activation='relu', name='third conv')(pool_2)

	conv_4 = Conv1D(filters=1024, kernel_size=3,  padding='valid',
					activation='relu', name='fourth conv')(conv_3)

	conv_5 = Conv1D(filters=1024, kernel_size=3,  padding='valid',
					activation='relu', name='fifth conv')(conv_4)

	conv_6 = Conv1D(filters=1024, kernel_size=3,  padding='valid',
					activation='relu', name='sixth conv')(conv_5)

	pool_6 = MaxPooling1D(pool_size=3, padding='valid', name='third pool')(conv_6)



					


	
	
	
	
	
	




