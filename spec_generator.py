import os
import numpy as np

def preprocess(paths):
	input_spec = np.zeros((1,512,128))
	output_spec = np.zeros((1,512,128))
	for path in paths:
		if  os.path.isdir(path):
			mixture, _ = librosa.load(path+'/mixture.wav',sr =8192)
			vocals, _ = librosa.load(path+'/vocals.wav',sr =8192)
			mixture_spec = np.abs(librosa.stft(mixture,n_fft=1024,hop_length=768))
			vocals_spec = np.abs(librosa.stft(vocals,n_fft=1024,hop_length=768))
			mixture_spec = mixture_spec[:512,:mixture_spec.shape[1]/128*128]
			vocals_spec = vocals_spec[:512,:vocals_spec.shape[1]/128*128]
			#something like (512, 2176)
			mixture_splitted = np.array(np.split(mixture_spec,axis=1))
			vocals_splitted = np.array(np.split(vocals_spec,axis=1))
			print "mixture_splitted.shape,vocals_splitted,shape:",mixture_splitted.shape,vocals_splitted,shape
			input_spec = np.concatenate((mixture_splitted,input_spec),axis=0)
			output_spec = np.concatenate((vocals_splitted,output_spec),axis=0)
			#normalize magnitude spec ?

	input_spec = input_spec[1:]
	output_spec = output_spec[1:]
	input_spec.reshape(input_spec.shape+(1,))
	output_spec.reshape(output_spec.shape+(1,))
	print "input_spec.shape,output_spec.shape:",input_spec.shape,output_spec.shape
	return input_spec,output_spec


global i
i = 0
def spec_generator(input_spec,output_spec):#musdb
	#for path in paths:
	while True:
		global i 
		print "generator global i:",i
		i += 1
		#i %= (len(input_spec)/128)
		yield (input_spec[i*128:(i+1)*128] ,output_spec[i*128:(i+1)*128])


