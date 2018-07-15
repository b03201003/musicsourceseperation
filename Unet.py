import os 
import sys
import numpy as np
#from tqdm import tqdm
import keras
from keras.models import Sequential,Model
from keras.layers import Multiply,Concatenate,Lambda,Input,Conv2D,Conv2DTranspose,UpSampling2D,MaxPooling2D,Dense, Dropout, Activation,BatchNormalization,LeakyReLU
from keras.optimizers import RMSprop, Adadelta,Nadam,SGD
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import regularizers
#from spec_generator import spec_generator
class Unet():
	def __init__(self,input_shape):#input_shape = (512,128,1)
		# adapt this if using `channels_first` image data format
		
		input_spec = Input(shape=input_shape)

		#encode part:
		encoded_spec = Conv2D(filters =16,kernel_size=(5,5),strides=2,padding='same')(input_spec)
		encoded_spec = BatchNormalization()(encoded_spec)
		encoded_1st = LeakyReLU(alpha = 0.2)(encoded_spec)

		encoded_spec = Conv2D(filters =32,kernel_size=(5,5),strides=2,padding='same')(encoded_1st)
		encoded_spec = BatchNormalization()(encoded_spec)
		encoded_2nd = LeakyReLU(alpha = 0.2)(encoded_spec)

		encoded_spec = Conv2D(filters =64,kernel_size=(5,5),strides=2,padding='same')(encoded_2nd)
		encoded_spec = BatchNormalization()(encoded_spec)
		encoded_3rd = LeakyReLU(alpha = 0.2)(encoded_spec)

		encoded_spec = Conv2D(filters =128,kernel_size=(5,5),strides=2,padding='same')(encoded_3rd)
		encoded_spec = BatchNormalization()(encoded_spec)
		encoded_4th = LeakyReLU(alpha = 0.2)(encoded_spec)

		encoded_spec = Conv2D(filters =256,kernel_size=(5,5),strides=2,padding='same')(encoded_4th)
		encoded_spec = BatchNormalization()(encoded_spec)
		encoded_5th = LeakyReLU(alpha = 0.2)(encoded_spec)

		encoded_spec = Conv2D(filters =512,kernel_size=(5,5),strides=2,padding='same')(encoded_5th)
		encoded_spec = BatchNormalization()(encoded_spec)
		encoded_result = LeakyReLU(alpha = 0.2)(encoded_spec)
		#decode part:
		decoded_spec = Conv2DTranspose(filters =256,kernel_size=(5,5),strides=2,padding='same')(encoded_result)
		decoded_spec = BatchNormalization()(decoded_spec)
		decoded_spec = Activation('relu')(decoded_spec)
		decoded_spec = Dropout(0.5)(decoded_spec)
		decoded_spec = Concatenate()([decoded_spec,encoded_5th])

		decoded_spec = Conv2DTranspose(filters =128,kernel_size=(5,5),strides=2,padding='same')(decoded_spec)
		decoded_spec = BatchNormalization()(decoded_spec)
		decoded_spec = Activation('relu')(decoded_spec)
		decoded_spec = Dropout(0.5)(decoded_spec)
		decoded_spec = Concatenate()([decoded_spec,encoded_4th])

		decoded_spec = Conv2DTranspose(filters =64,kernel_size=(5,5),strides=2,padding='same')(decoded_spec)
		decoded_spec = BatchNormalization()(decoded_spec)
		decoded_spec = Activation('relu')(decoded_spec)
		decoded_spec = Dropout(0.5)(decoded_spec)
		decoded_spec = Concatenate()([decoded_spec,encoded_3rd])

		decoded_spec = Conv2DTranspose(filters =32,kernel_size=(5,5),strides=2,padding='same')(decoded_spec)
		decoded_spec = BatchNormalization()(decoded_spec)
		decoded_spec = Activation('relu')(decoded_spec)
		#decoded_spec = Dropout(0.5)(decoded_spec)
		decoded_spec = Concatenate()([decoded_spec,encoded_2nd])

		decoded_spec = Conv2DTranspose(filters =16,kernel_size=(5,5),strides=2,padding='same')(decoded_spec)
		decoded_spec = BatchNormalization()(decoded_spec)
		decoded_spec = Activation('relu')(decoded_spec)
		#decoded_spec = Dropout(0.5)(decoded_spec)
		decoded_spec = Concatenate()([decoded_spec,encoded_1st])

		decoded_spec = Conv2DTranspose(filters =1,kernel_size=(5,5),strides=2,padding='same')(decoded_spec)
		decoded_spec = BatchNormalization()(decoded_spec)
		decoded_mask = Activation('sigmoid')(decoded_spec)
		#decoded_spec = Dropout(0.5)(decoded_spec)
		#decoded_spec = Concatenate()([decoded_spec,encoded_5th])
		output_spec = Multiply()([input_spec,decoded_mask])   

		self.model = Model(inputs= input_spec ,outputs=output_spec)
		nadam=Nadam(lr=0.002,beta_1=0.9,beta_2=0.999,epsilon=1e-08,schedule_decay=0.004)
		#self.model = Model(input_img, decoded)
		self.model.summary()
		self.model.compile(optimizer=nadam, loss='mae',metrics=['mse','mae','acc'])
		print self.model.get_config()
		
		#self.model.compile(optimizer=nadam,loss='mean_squared_error',metrics=['mae','accuracy'])
		
		#dropout or bn?
		#what range of value does the output must be?
	def model(self):
		return self.model




if __name__ == '__main__': 
	model = Unet((512,128,1))
	#model.train()


