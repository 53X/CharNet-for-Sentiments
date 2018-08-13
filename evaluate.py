import keras
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
import math



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





