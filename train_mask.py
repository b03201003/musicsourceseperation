import os 
import sys
import numpy as np
#from tqdm import tqdm
import keras
#from maskLSTM import LSTM
#from maskLSTMlight2 import LSTM
#from maskLSTMlight import LSTM
from maskBiLSTM import LSTM
#from maskBiLSTM2 import LSTM
import json
from myModelCheckpoint import *
from keras.callbacks import EarlyStopping

#ISMIR 2018 2017
#BiLSTM on spec model , attention, BN layer
#data preprocessing 

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

model = LSTM()

epochs = 5 
batch_size = 100 
frame_wedth = 15
freq_bin = 1025
#DEMO
#be careful of MemoryError!!!
#use del to close the np.array
#train_accompaniment=np.load('./data/train_accompaniment.npy')
#train_accompaniment = train_accompaniment[rand_index]
#weight_file  =  './mask001.hdf5'  
weight_file = sys.argv[2]
#if not os.path.exists(os.path.dirname(weight_file)):
#	os.makedirs(os.path.dirname(weight_file))   
  
if os.path.isfile(weight_file):
	print('trained weight exists!')
	model.load_weights(weight_file)
	print('complete load weights!!')
checkpointer = myModelCheckpoint(filepath=weight_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10) 

x_train = np.load('../CDAE/maskdata/train_maskX.npy')
x_valid = np.load('../CDAE/maskdata/valid_maskX.npy')
# x_train = x_train[:(len(x_train)/batch_size)*batch_size]
# x_valid = x_valid[:(len(x_valid)/batch_size)*batch_size]

train_pad_len = batch_size - (len(x_train)%batch_size)
valid_pad_len = batch_size - (len(x_valid)%batch_size)
x_train = np.concatenate((x_train,np.zeros([train_pad_len,frame_wedth,freq_bin])),axis=0)
x_valid = np.concatenate((x_valid,np.zeros([valid_pad_len,frame_wedth,freq_bin])),axis=0)
# shape = list(x_train.shape) 
# shape.append(1)
# x_train = x_train.reshape(shape)
# shape = list(x_valid.shape) 
# shape.append(1)
# x_valid = x_valid.reshape(shape)

y_train = np.load('../CDAE/maskdata/train_maskY.npy')
y_valid = np.load('../CDAE/maskdata/valid_maskY.npy')
shape = list(y_train.shape) 
shape.append(1)
y_train = y_train.reshape(shape)
shape = list(y_valid.shape) 
shape.append(1)
y_valid = y_valid.reshape(shape)


y_train = np.concatenate((y_train,np.zeros([train_pad_len,frame_wedth,1])),axis=0)
y_valid = np.concatenate((y_valid,np.zeros([valid_pad_len,frame_wedth,1])),axis=0)
# y_train = y_train[:(len(y_train)/batch_size)*batch_size]
# y_valid = y_valid[:(len(y_valid)/batch_size)*batch_size]
# shape = list(y_train.shape) 
# shape.append(1)
# y_train = y_train.reshape(shape)
# shape = list(y_valid.shape) 
# shape.append(1)
# y_valid = y_valid.reshape(shape)
print(x_train.shape,y_train.shape)
print(x_valid.shape,y_valid.shape)


# x_test= np.load('./data/test_vocals.npy')
# shape = list(x_test.shape) 
# shape.append(1)
# x_test = x_test.reshape(shape)
# y_test= x_test
# print x_test.shape,y_test.shape



for i  in range(100):
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> training : ' , str(i+1) ,'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


	#print "first part: (voice,voice)"
	# train_vocals = np.load('./data/train_vocals.npy')
	# #rand_index  = np.random.choice(train_vocals.shape[0] ,int(train_vocals.shape[0]),replace = False) #only need calculate it once
	# #train_vocals = train_vocals[rand_index]
	# x_train = train_vocals
	# y_train = train_vocals
	# del train_vocals
	# x_valid = np.load('./data/valid_vocals.npy')
	# y_valid = np.load('./data/valid_vocals.npy')

	# shape = list(x_train.shape) 
	# shape.append(1)
	# x_train = x_train.reshape(shape)
	# shape = list(y_train.shape) 
	# shape.append(1)
	# y_train = y_train.reshape(shape)
	# shape = list(x_valid.shape) 
	# shape.append(1)
	# x_valid = x_valid.reshape(shape)
	# shape = list(y_valid.shape) 
	# shape.append(1)
	# y_valid = y_valid.reshape(shape)
	# print x_train.shape,y_train.shape
	# print x_valid.shape,y_valid.shape

	model.train(x_train,y_train,x_valid,y_valid,checkpointer,early_stopping,epochs,batch_size)
	#learning rate reduce...by a factor of 10 when the values of the cost function do not decrease on the validation set for 3 consecutive epochs
	# model.evaluate(x_test,y_test,batch_size)
	# if i%5==0:
	# 	y_predict = model.predict(x_test,batch_size)
	# 	print "y_predict.shape:",y_predict.shape
	# 	np.save('predict001.npy',y_predict)

	# del x_train
	# del y_train
	# del x_valid 
	# del y_valid 
	#predict + istft
	#



	# print "second part: (mix,voice)"

	# train_vocals = np.load('./data/train_vocals.npy')
	# #rand_index  = np.random.choice(train_vocals.shape[0] ,int(train_vocals.shape[0]),replace = False) #only need calculate it once
	# #train_vocals = train_vocals[rand_index]
	# train_mixture = np.load('./data/train_mixture.npy')
	# #train_mixture = train_mixture[rand_index]
	# print train_vocals.shape,train_mixture.shape


	# x_train = train_mixture
	# del train_mixture
	# shape = list(x_train.shape) 
	# shape.append(1)
	# x_train = x_train.reshape(shape)
	# y_train = train_vocals
	# del train_vocals
	# shape = list(y_train.shape) 
	# shape.append(1)
	# y_train = y_train.reshape(shape)
	# print x_train.shape,y_train.shape
	# x_valid = np.load('./data/valid_mixture.npy')
	# y_valid = np.load('./data/valid_vocals.npy')
	# shape = list(x_valid.shape) 
	# shape.append(1)
	# x_valid = x_valid.reshape(shape)
	# shape = list(y_valid.shape) 
	# shape.append(1)
	# y_valid = y_valid.reshape(shape)
	# print x_valid.shape,y_valid.shape
	# model.train(x_train,y_train,x_valid,y_valid,checkpointer,early_stopping,epochs,batch_size)
	# del x_train
	# del y_train
	# del x_valid 
	# del y_valid 

	# print "third part: (accom,zero)"

	# x_train = np.load('./data/train_accompaniment.npy')
	# shape = list(x_train.shape) 
	# shape.append(1)
	# x_train = x_train.reshape(shape)
	# y_train = np.zeros(list(x_train.shape))
	# print x_train.shape,y_train.shape
	# x_valid = np.load('./data/valid_accompaniment.npy')
	# shape = list(x_valid.shape) 
	# shape.append(1)
	# x_valid = x_valid.reshape(shape)
	# y_valid = np.zeros(list(x_valid.shape))
	# print x_valid.shape,y_valid.shape
	# model.train(x_train,y_train,x_valid,y_valid,checkpointer,early_stopping,epochs,batch_size)
	# del x_train
	# del y_train
	# del x_valid 
	# del y_valid


#evaluate
"""
x_test= np.load('./data/test_mixture.npy')
y_test= np.load('./data/test_vocals.npy')
shape = list(x_test.shape) 
shape.append(1)
x_test = x_test.reshape(shape)
shape = list(y_test.shape) 
shape.append(1)
y_test = y_test.reshape(shape)
print x_test.shape,y_test.shape
model.evaluate(x_test,y_test,batch_size)
"""

