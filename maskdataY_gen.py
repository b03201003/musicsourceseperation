import os 
import sys
import numpy as np
from tqdm import tqdm
import librosa

source_data_path = '../CDAE/musdb18/'
training_data_path = '../CDAE/maskdata/'
frame_wedth = 15
freq_bin = 1025

mask_label = np.zeros([1,frame_wedth])


for traintest in os.listdir(source_data_path):
	
	if traintest=='train' or traintest == 'test':#test -> 50% valid, 50% test
		print "Now in :",source_data_path+traintest
		print "os.listdir(source_data_path+traintest):",os.listdir(source_data_path+traintest)
		for song in os.listdir(source_data_path+traintest): 
			if os.path.isdir(source_data_path+traintest+'/'+song):
				song+='/'
				print "Now in :",source_data_path+traintest+'/'+song
				vocals, sr = librosa.load(source_data_path+traintest+'/'+song+'vocals.wav') #default: sr=22050
				print "vocals.shape:",vocals.shape
				vocals_spec = np.abs(librosa.core.spectrum.stft(vocals, n_fft=2048))#default: hop_length=512, win_length=2048
				print "vocals_spec.shape:",vocals_spec.shape
				#for i in range(vocals_spec.shape[1]):
					#print "spec sum:",np.sum(vocals_spec.T[i])
				#break
				time_length = vocals_spec.shape[1]
				print "vocals_spec.shape:",vocals_spec.shape
				vocals_mask = np.zeros([time_length])
				for i in range(time_length):
					if np.sum(vocals_spec.T[i])>100.0:
						vocals_mask[i] = 1.0
				print"vocals_mask:",vocals_mask
				vocals_mask = np.concatenate((vocals_mask[:frame_wedth*(time_length/frame_wedth)],vocals_mask[-frame_wedth:]),axis=0)

				vocals_mask = np.array(np.split(vocals_mask,(time_length/frame_wedth)+1))	
				print "vocals_mask.shape:",vocals_mask.shape
				mask_label = np.concatenate((mask_label,vocals_mask),axis=0)
				print "mask_label.shape:",mask_label.shape

        np.save(training_data_path+traintest+'_maskY.npy',mask_label[1:])
        mask_label = np.zeros([1,frame_wedth])