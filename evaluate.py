import keras
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
import math
from prep_data import structure_data
from model import character_model
from sklearn.model_selection import train_test_split 

def step_decay(epoch):

	drop = 0.5
	init_learn_rate = 0.01
	epoch_drop = 3

	learning_rate = init_learn_rate*(drop**math.floor(epoch/epoch_drop))

	return learning_rate


learning_rate_scheduler = LearningRateScheduler(step_decay, verbose=1)
modelcheckpoint = ModelCheckpoint('saved_model.h5', save_best_only=True,
								   monitor='val_acc', mode='max', period=3,
								  verbose=1)
earlystopping = EarlyStopping(monitor='val_acc', mode='max', patience=5, verbose=1)

data, labels = structure_data()
x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.94,
													test_size=0.06, stratify=labels)

model = character_model(classification=4)

model.fit(x_train, y_train, batch_size=128, epochs=200,
		  callbacks=[modelcheckpoint, earlystopping, learning_rate_scheduler])

print("ACCURACY ON TEST SET : ", model.evaluate(x_test, y_test))



