import os 
import sys
import numpy as np
from tqdm import tqdm
import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,BatchNormalization,Flatten,Reshape,Input,CuDNNLSTM,TimeDistributed
from keras.optimizers import RMSprop, Adadelta,Nadam,SGD
from keras.layers.convolutional import Conv2D,UpSampling2D,MaxPooling2D
from keras.callbacks import EarlyStopping
from keras import backend as K
def custom_loss(y_true,y_pred):
	loss = K.sum(K.square(y_true-y_pred))
	return loss 
class LSTM():
	def __init__(self,input_shape=(100,15,1)):
		
		#input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data format
		self.model = Sequential()
		self.model.add(TimeDistributed(Dense(100,activation = 'relu'), batch_input_shape=input_shape))
		self.model.add(CuDNNLSTM(50,return_sequences=True,stateful=True))
		self.model.add(Activation('relu'))
		self.model.add(CuDNNLSTM(100,return_sequences=True,stateful=True))
		self.model.add(Activation('relu'))
		self.model.add(TimeDistributed(Dense(1,activation = 'sigmoid')))
		nadam=Nadam(lr=0.002,beta_1=0.9,beta_2=0.999,epsilon=1e-08,schedule_decay=0.004)
		#self.model = Model(input_img, decoded)
		self.model.compile(optimizer=nadam, loss=custom_loss,metrics=['mse','mae'])

		
		#self.model.compile(optimizer=nadam,loss='mean_squared_error',metrics=['mae','accuracy'])
		self.model.summary()
		#dropout or bn?
		#what range of value does the output must be?
	def load_weights(self,weights):
		self.model.load_weights(weights)

	def train(self,x_train,y_train,x_valid,y_valid,checkpointer,early_stopping,epochs=20,batch_size=64):
		self.model.fit(x_train,y_train,validation_data=(x_valid,y_valid),epochs=epochs,batch_size=batch_size,callbacks=[checkpointer,early_stopping])
	def evaluate(self,x_test,y_test,batch_size=100):
		print "\nevaluate:",self.model.evaluate(x_test,y_test,batch_size=batch_size)

	def predict(self,x_test,batch_size):
		y_predict = self.model.predict(x_test,batch_size=batch_size)
		return y_predict 



if __name__ == '__main__': 
	model = LSTM()
	#model.train()


