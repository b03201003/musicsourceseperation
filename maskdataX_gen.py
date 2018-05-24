import os 
import sys
import numpy as np
from tqdm import tqdm
import librosa

source_data_path = '../CDAE/musdb18/'
training_data_path = '../CDAE/maskdata/'
frame_wedth = 15
freq_bin = 1025

mixture_data = np.zeros([1,frame_wedth,freq_bin])


for traintest in os.listdir(source_data_path):
	
	if traintest=='train' or traintest == 'test':#test -> 50% valid, 50% test
		print "Now in :",source_data_path+traintest
		print "os.listdir(source_data_path+traintest):",os.listdir(source_data_path+traintest)
		for song in os.listdir(source_data_path+traintest): 
			if os.path.isdir(source_data_path+traintest+'/'+song):
				song+='/'
				print "Now in :",source_data_path+traintest+'/'+song
				mixture, sr = librosa.load(source_data_path+traintest+'/'+song+'mixture.wav') #default: sr=22050

				mixture_spec = np.abs(librosa.core.spectrum.stft(mixture, n_fft=2048))#default: hop_length=512, win_length=2048

				time_length = mixture_spec.shape[1]

				#mixture_spec = mixture_spec.T[:frame_wedth*(time_length/frame_wedth)]
				mixture_spec = mixture_spec.T

				print mixture_spec.shape
				# if time_length<15:
				# 	continue
				for i in range(time_length):
					print"i:",i
					if i<7:
						padding = np.zeros([1,7-i,freq_bin])  
						cur_spec = np.concatenate((padding,mixture_spec[:i+8].reshape(1,i+8,freq_bin)),axis=1)
					elif i> time_length-8:  #last i:time_length-1
						padding = np.zeros([1, 8-time_length+i ,freq_bin]) 
						cur_spec = np.concatenate((mixture_spec[i-7:].reshape(1,time_length-i+7,freq_bin),padding),axis=1)
					else:
						cur_spec = mixture_spec[i-7:i+8].reshape(1,frame_wedth,freq_bin)
					print"cur_spec.shape:",cur_spec.shape
					mixture_data = np.concatenate((mixture_data,cur_spec),axis=0)
					print "mixture_data.shape:"mixture_data.shape


        #np.save(training_data_path+traintest+'_mixture.npy',mixture_data[1:])
        np.save(training_data_path+traintest+'_mixture.npy',vocals_data[1:])
        #np.save(training_data_path+traintest+'_accompaniment.npy',accompaniment_data[1:])
        #mixture_data = np.zeros([1,frame_wedth,freq_bin])
        vocals_data = np.zeros([1,frame_wedth,freq_bin])
        #accompaniment_data = np.zeros([1,frame_wedth,freq_bin])