import keras
from keras.layers import Conv1D, MaxPooling1D, Dense, Input, Embedding, Dropout, Flatten
from keras.optimizers import SGD
from keras.initializers import RandomNormal
from keras.models import Model


def character_model(classification, vocab_size=69, maxlen=1014):

	optimizer = SGD(lr=0.01, momentum=0.9)

	init = RandomNormal(mean=0.0, stddev=0.02)

	input_layer = Input(shape=(maxlen, ), name='sentence_input')

	embedding = Embedding(input_dim = vocab_size+1, output_dim = 70,
						  name='character_embedding')(input_layer)

	conv_1 = Conv1D(filters=1024, kernel_size=7,  padding='valid',
					activation='relu',kernel_initializer=init ,name='first_conv')(embedding)

	pool_1 = MaxPooling1D(pool_size=3, padding='valid', name='first_pool')(conv_1)

	conv_2 = Conv1D(filters=1024, kernel_size=7,  padding='valid',
					activation='relu',kernel_initializer=init, name='second_conv')(pool_1)

	pool_2 = MaxPooling1D(pool_size=3, padding='valid', name='second_pool')(conv_2)

	conv_3 = Conv1D(filters=1024, kernel_size=3,  padding='valid',
					activation='relu',kernel_initializer=init, name='third_conv')(pool_2)

	conv_4 = Conv1D(filters=1024, kernel_size=3,  padding='valid',
					activation='relu',kernel_initializer=init, name='fourth_conv')(conv_3)

	conv_5 = Conv1D(filters=1024, kernel_size=3,  padding='valid',
					activation='relu',kernel_initializer=init, name='fifth_conv')(conv_4)

	conv_6 = Conv1D(filters=1024, kernel_size=3,  padding='valid',
					activation='relu',kernel_initializer=init, name='sixth_conv')(conv_5)

	pool_6 = MaxPooling1D(pool_size=3, padding='valid', name='third_pool')(conv_6)

	flattened = Flatten()(pool_6)

	fully_conn_1 = Dense(2048, activation='relu')(flattened)

	fully_conn_1 = Dropout(0.5)(fully_conn_1)

	fully_conn_2 = Dense(2048, activation='relu')(fully_conn_1)

	fully_conn_2 = Dropout(0.5)(fully_conn_2)

	output = Dense(classification, activation='softmax')(fully_conn_2)

	model = Model(inputs = input_layer, outputs = output)

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	return model

					


	
	
	
	
	
	




