import os 
import sys
import numpy as np
from tqdm import tqdm
import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,BatchNormalization,Flatten,Reshape,Input
from keras.optimizers import RMSprop, Adadelta,Nadam,SGD
from keras.layers.convolutional import Conv2D,UpSampling2D,MaxPooling2D
from keras.callbacks import EarlyStopping
from keras import backend as K
def custom_loss(y_true,y_pred):
	loss = K.sum(K.square(y_true-y_pred))
	return loss 

class CDAE():
	def __init__(self,input_shape=(15,1025,1)):
		self.model = Sequential()  #input size:(None,15,1025)
		self.model.add(Conv2D(12,(3,3),input_shape=input_shape,padding='same',kernel_initializer='zeros'))
		#self.model.add(BatchNormalization())
		#self.model.add(Dropout(0.2))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D((3,5)))
		self.model.add(Conv2D(20,(3,3),padding='same',kernel_initializer='zeros'))
		#self.model.add(BatchNormalization())
		#self.model.add(Dropout(0.2))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D((1,5)))
		self.model.add(Conv2D(30,(3,3),padding='same',kernel_initializer='zeros'))
		#self.model.add(BatchNormalization())
		#self.model.add(Dropout(0.2))
		self.model.add(Activation('relu'))
		self.model.add(Conv2D(40,(3,3),padding='same',kernel_initializer='zeros'))
		#self.model.add(BatchNormalization())
		#self.model.add(Dropout(0.2))
		self.model.add(Activation('relu'))
		self.model.add(Conv2D(30,(3,3),padding='same',kernel_initializer='zeros'))
		#self.model.add(BatchNormalization())
		#self.model.add(Dropout(0.2))
		self.model.add(Activation('relu'))
		self.model.add(Conv2D(20,(3,3),padding='same',kernel_initializer='zeros'))
		#self.model.add(BatchNormalization())
		#self.model.add(Dropout(0.2))
		self.model.add(Activation('relu'))
		self.model.add(UpSampling2D((1,5)))
		self.model.add(Conv2D(12,(3,3),padding='same',kernel_initializer='zeros'))
		#self.model.add(BatchNormalization())
		#self.model.add(Dropout(0.2))
		self.model.add(Activation('relu'))
		self.model.add(UpSampling2D((3,5)))
		self.model.add(Conv2D(1,(3,3),padding='same',kernel_initializer='zeros'))
		#self.model.add(BatchNormalization())
		#self.model.add(Dropout(0.2))
		self.model.add(Activation('relu'))
		nadam=Nadam(lr=0.002,beta_1=0.9,beta_2=0.999,epsilon=1e-08,schedule_decay=0.004)
		self.model.compile(optimizer=nadam,loss=custom_loss,metrics=['mse','mae','accuracy'])
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
	model = CDAE()
	#model.train()


