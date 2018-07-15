from __future__ import print_function
import numpy as np
#from tqdm import tqdm
import keras
import os
import sys
from spec_generator import *
from Unet import Unet
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
from glob import glob

train_paths = glob('/home/u/b03201003/CDAE/musdb18/train/*')
test_paths = glob('/home/u/b03201003/CDAE/musdb18/test/*')


model = Unet(input_shape=(512,128,1)).model
weight_file  =  './Unet.hdf5'   
if os.path.isfile(weight_file):
	print ('trained weight exists!')
	model.load_weights(weight_file)
	print ('complete load weights!!')
#"""
preprocess2(train_paths,'train')
preprocess2(test_paths,'test')
preprocess3(train_paths,'train')
preprocess3(test_paths,'test')
#"""
batch_size = 32
if sys.argv[1]== 'train':
	train_input_spec,train_output_spec = preprocess(train_paths,'train')
	train_input_spec = train_input_spec.reshape(train_input_spec.shape+(1,))
	train_output_spec = train_output_spec.reshape(train_output_spec.shape+(1,))
	evaluate_input_spec,evaluate_output_spec = preprocess(test_paths,'test')
	evaluate_input_spec = evaluate_input_spec.reshape(evaluate_input_spec.shape+(1,))
	evaluate_output_spec = evaluate_output_spec.reshape(evaluate_output_spec.shape+(1,))
	print("train_input_spec.shape,train_output_spec.shape,evaluate_input_spec.shape,evaluate_output_spec.shape:",train_input_spec.shape,train_output_spec.shape,evaluate_input_spec.shape,evaluate_output_spec.shape)

	TrainEpochs = int(sys.argv[3])
	for i in range(TrainEpochs):
		History = model.fit_generator(spec_generator(train_input_spec,train_output_spec,batch_size),steps_per_epoch=len(train_input_spec)/batch_size ,epochs=1)
		#History = CrossModalityCNNmodel.fit(x=input_list,y=label,batch_size=batch_size,epochs=1,validation_split=0.2)
	#CrossModalityCNNmodel.save('myModel.h5')
		model.save_weights('./Unet.hdf5', overwrite=True)
		loss = model.evaluate_generator(spec_generator(evaluate_input_spec,evaluate_output_spec,batch_size),steps=len(evaluate_input_spec)/batch_size)
		print ("{}-th epoch testing loss:{}\n".format(i,loss))
elif sys.argv[1]== 'predict':
	#predict_path = test_paths[0]
	batch_size = 4
	predict_path = './mixture.wav'
	test_mix,sr = librosa.load(predict_path,sr =8192)
	D = librosa.stft(test_mix,n_fft=1024,hop_length=768)
	test_spec = np.abs(D)
	padded_spec = np.concatenate((test_spec[:512,:(test_spec.shape[1])],np.zeros((512,(128-(test_spec.shape[1]%128))))),axis=1)
	test_input_spec = np.array(np.split(padded_spec,padded_spec.shape[1]//128,axis=1))
	
	test_pad_len = batch_size - (len(test_input_spec)%batch_size)
	test_input_spec = np.concatenate((test_input_spec,np.zeros([test_pad_len,512,128])),axis=0)
	test_input_spec = test_input_spec.reshape(test_input_spec.shape+(1,))
	print("test_input_spec.shape:",test_input_spec.shape)

	predict_outputs = model.predict(test_input_spec,batch_size = batch_size)
	print ("predict_outputs.shape:",predict_outputs.shape)
	predict_outputs = predict_outputs.reshape((-1,512,128))
	predict_outputs = np.concatenate(tuple([spec for spec in predict_outputs]),axis=-1)
	#predict_outputs = predict_outputs.reshape((512,-1))
	#Becareful using reshape!!!!!
	print ("predict_outputs.shape:",predict_outputs.shape)
	np.save('Unet_prediction.npy',predict_outputs)

	#istft 
	phase = np.angle(D)
	print("D.shape:",D.shape)
	output = np.zeros(D.shape)
	output[:512,:] = predict_outputs[:,:output.shape[1]]
	i = 0+1j
	c = np.multiply( output,np.exp(i*phase))
	y_wav = librosa.istft(c,hop_length=768)# argument???
	print ("y_wav:",y_wav)
	librosa.output.write_wav('./mixture_test.wav',y_wav,sr)
	print ("len(y_wav),len(test_mix):",len(y_wav),len(test_mix))
	if len(test_mix) > len(y_wav):
	    test_mix = test_mix[:len(y_wav)]
	else : 
	    y_wav = y_wav[:len(test_mix)]
	accompany = np.subtract(test_mix,y_wav) 
	print("mix sum,vocal sum,accompany sum:",np.sum(test_mix),np.sum(y_wav),np.sum(accompany))
	librosa.output.write_wav('./mixture_accompany_test.wav',accompany,sr)

	# 1. Reference "Singing voice separation with deep U-Net convolutional networks"
	# 2. style transfer
	# 3. training strategy







