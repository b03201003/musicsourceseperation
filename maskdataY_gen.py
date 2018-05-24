import os 
import sys
import numpy as np
from tqdm import tqdm
import librosa

source_data_path = '../CDAE/musdb18/'
training_data_path = '../CDAE/maskdata/'
frame_wedth = 15
freq_bin = 1025

mask_label = np.zeros([1])


for traintest in os.listdir(source_data_path):
	
	if traintest=='train' or traintest == 'test':#test -> 50% valid, 50% test
		print "Now in :",source_data_path+traintest
		print "os.listdir(source_data_path+traintest):",os.listdir(source_data_path+traintest)
		for song in os.listdir(source_data_path+traintest): 
			if os.path.isdir(source_data_path+traintest+'/'+song):
				song+='/'
				print "Now in :",source_data_path+traintest+'/'+song
				mixture, sr = librosa.load(source_data_path+traintest+'/'+song+'mixture.wav') #default: sr=22050
				print "mixture.shape:",mixture.shape
				mixture_spec = np.abs(librosa.core.spectrum.stft(mixture, n_fft=2048))#default: hop_length=512, win_length=2048
				print "mixture_spec.shape:",mixture_spec.shape
				for i in range(mixture_spec.shape[1]):
					print "spec sum:",np.sum(mixture_spec.T[i])

				break
				# time_length = mixture_spec.shape[1]

				# #mixture_spec = mixture_spec.T[:frame_wedth*(time_length/frame_wedth)]
				# #last = mixture_spec.T[-frame_wedth:]

				# mixture_spec = np.concatenate((mixture_spec.T[:frame_wedth*(time_length/frame_wedth)],mixture_spec.T[-frame_wedth:]),axis=0)
				# print "mixture_spec.shape:",mixture_spec.shape
				# mixture_spec = np.array(np.split(mixture_spec,(time_length/frame_wedth)+1))	
				# print "mixture_spec.shape:",mixture_spec.shape
				# mask_data = np.concatenate((mask_data,mixture_spec),axis=0)
				# print "mask_data.shape:",mask_data.shape

    #     np.save(training_data_path+traintest+'_maskX.npy',mask_data[1:])
    #     mask_data = np.zeros([1,frame_wedth,freq_bin])