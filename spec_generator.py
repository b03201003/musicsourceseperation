from __future__ import print_function
import os
import numpy as np
import librosa
def preprocess(paths,stage): #mixture->voice
	if os.path.isfile(stage+'_Unet_input.npy') and os.path.isfile(stage+'_Unet_output.npy'):
		input_spec = np.load(stage+'_Unet_input.npy')
		output_spec = np.load(stage+'_Unet_output.npy') 
	else:
		input_spec = np.zeros((1,512,128))
		output_spec = np.zeros((1,512,128))
		for path in paths:
			if  os.path.isdir(path):
				mixture, _ = librosa.load(path+'/mixture.wav',sr =8192)
				vocals, _ = librosa.load(path+'/vocals.wav',sr =8192)
				mixture_spec = np.abs(librosa.stft(mixture,n_fft=1024,hop_length=768))
				vocals_spec = np.abs(librosa.stft(vocals,n_fft=1024,hop_length=768))
				#print("mixture_spec.shape,vocals_spec.shape:",mixture_spec.shape,vocals_spec.shape)
				mixture_spec = np.concatenate((mixture_spec[:512,:(mixture_spec.shape[1])],np.zeros((512,(128-(mixture_spec.shape[1]%128))))),axis=1)
				vocals_spec = np.concatenate((vocals_spec[:512,:(vocals_spec.shape[1])],np.zeros((512,(128-(vocals_spec.shape[1]%128))))),axis=1)

				print("mixture_spec.shape,vocals_spec.shape:",mixture_spec.shape,vocals_spec.shape)
				#something like (512, 2176)
				mixture_splitted = np.array(np.split(mixture_spec,mixture_spec.shape[1]//128,axis=1))
				vocals_splitted = np.array(np.split(vocals_spec,vocals_spec.shape[1]//128,axis=1))
				#print(mixture_splitted,vocals_splitted)
				print("mixture_splitted.shape,vocals_splitted,shape:",mixture_splitted.shape,vocals_splitted.shape)
				input_spec = np.concatenate((mixture_splitted,input_spec),axis=0)
				output_spec = np.concatenate((vocals_splitted,output_spec),axis=0)
				#normalize magnitude spec ?

		input_spec = input_spec[1:]
		output_spec = output_spec[1:]
		input_spec.reshape(input_spec.shape+(1,))
		output_spec.reshape(output_spec.shape+(1,))
		np.save(stage+'_Unet_input.npy',input_spec)
		np.save(stage+'_Unet_output.npy',output_spec)
	print("input_spec.shape,output_spec.shape:",input_spec.shape,output_spec.shape)
	return input_spec,output_spec

def preprocess2(paths,stage): #accompany->zeros
	if os.path.isfile(stage+'_Unet_input2.npy') and os.path.isfile(stage+'_Unet_output2.npy'):
		input_spec = np.load(stage+'_Unet_input2.npy')
		output_spec = np.load(stage+'_Unet_output2.npy') 
	else:
		input_spec = np.zeros((1,512,128))
		output_spec = np.zeros((1,512,128))
		for path in paths:
			if  os.path.isdir(path):
				accompaniment, _ = librosa.load(path+'/accompaniment.wav',sr =8192)
				#vocals, _ = librosa.load(path+'/vocals.wav',sr =8192)
				accompaniment_spec = np.abs(librosa.stft(accompaniment,n_fft=1024,hop_length=768))
				#vocals_spec = np.abs(librosa.stft(vocals,n_fft=1024,hop_length=768))
				#print("mixture_spec.shape,vocals_spec.shape:",mixture_spec.shape,vocals_spec.shape)
				accompaniment_spec = np.concatenate((accompaniment_spec[:512,:(accompaniment_spec.shape[1])],np.zeros((512,(128-(accompaniment_spec.shape[1]%128))))),axis=1)
				#vocals_spec = np.concatenate((vocals_spec[:512,:(vocals_spec.shape[1])],np.zeros((512,(128-(vocals_spec.shape[1]%128))))),axis=1)

				print("accompaniment_spec.shape:",accompaniment_spec.shape)
				#something like (512, 2176)
				accompaniment_splitted = np.array(np.split(accompaniment_spec,accompaniment_spec.shape[1]//128,axis=1))
				#vocals_splitted = np.array(np.split(vocals_spec,vocals_spec.shape[1]//128,axis=1))
				#print(mixture_splitted,vocals_splitted)
				print("accompaniment_splitted.shape:",accompaniment_splitted.shape)
				input_spec = np.concatenate((accompaniment_splitted,input_spec),axis=0)
				#output_spec = np.concatenate((vocals_splitted,output_spec),axis=0)
				#normalize magnitude spec ?

		input_spec = input_spec[1:]
		#output_spec = output_spec[1:]
		input_spec.reshape(input_spec.shape+(1,))
		#output_spec.reshape(output_spec.shape+(1,))
		output_spec = np.zeros(output_spec.shape)
		np.save(stage+'_Unet_input2.npy',input_spec)
		np.save(stage+'_Unet_output2.npy',output_spec)
	print("input_spec2.shape,output_spec2.shape:",input_spec.shape,output_spec.shape)
	return input_spec,output_spec

def preprocess3(paths,stage):#voice->voice
	if os.path.isfile(stage+'_Unet_input3.npy') and os.path.isfile(stage+'_Unet_output3.npy'):
		input_spec = np.load(stage+'_Unet_input3.npy')
		output_spec = np.load(stage+'_Unet_output3.npy') 
	else:
		input_spec = np.zeros((1,512,128))
		output_spec = np.zeros((1,512,128))
		for path in paths:
			if  os.path.isdir(path):
				#mixture, _ = librosa.load(path+'/mixture.wav',sr =8192)
				vocals, _ = librosa.load(path+'/vocals.wav',sr =8192)
				#mixture_spec = np.abs(librosa.stft(mixture,n_fft=1024,hop_length=768))
				vocals_spec = np.abs(librosa.stft(vocals,n_fft=1024,hop_length=768))
				#print("mixture_spec.shape,vocals_spec.shape:",mixture_spec.shape,vocals_spec.shape)
				#mixture_spec = np.concatenate((mixture_spec[:512,:(mixture_spec.shape[1])],np.zeros((512,(128-(mixture_spec.shape[1]%128))))),axis=1)
				vocals_spec = np.concatenate((vocals_spec[:512,:(vocals_spec.shape[1])],np.zeros((512,(128-(vocals_spec.shape[1]%128))))),axis=1)

				print("vocals_spec.shape:",vocals_spec.shape)
				#something like (512, 2176)
				#mixture_splitted = np.array(np.split(mixture_spec,mixture_spec.shape[1]//128,axis=1))
				vocals_splitted = np.array(np.split(vocals_spec,vocals_spec.shape[1]//128,axis=1))
				#print(mixture_splitted,vocals_splitted)
				print("vocals_splitted,shape:",vocals_splitted.shape)
				input_spec = np.concatenate((vocals_splitted,input_spec),axis=0)
				#output_spec = np.concatenate((vocals_splitted,output_spec),axis=0)
				#normalize magnitude spec ?

		input_spec = input_spec[1:]
		#output_spec = output_spec[1:]
		input_spec.reshape(input_spec.shape+(1,))
		#output_spec.reshape(output_spec.shape+(1,))
		output_spec = input_spec
		np.save(stage+'_Unet_input3.npy',input_spec)
		np.save(stage+'_Unet_output3.npy',output_spec)
	print("input_spec3.shape,output_spec3.shape:",input_spec.shape,output_spec.shape)
	return input_spec,output_spec





global i

def spec_generator(input_spec,output_spec,batch_size):#musdb
	#for path in paths:
	global i 
	i = 0
	while True:
		
		print ("\ngenerator global i:",i)
		yield (input_spec[i*batch_size:(i+1)*batch_size] ,output_spec[i*batch_size:(i+1)*batch_size])
		i += 1
		#i %= 58
		#print(len(input_spec)//batch_size)
		i %= (len(input_spec)//batch_size)


