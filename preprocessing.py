import os 
import sys
import numpy as np
from tqdm import tqdm
import librosa

source_data_path = './musdb18/'
training_data_path = './data/'
frame_wedth = 15
freq_bin = 1025

mixture_data = np.zeros([1,frame_wedth,freq_bin])
vocals_data = np.zeros([1,frame_wedth,freq_bin])
accompaniment_data = np.zeros([1,frame_wedth,freq_bin])

for traintest in os.listdir(source_data_path):
	
	if traintest=='train' or traintest == 'test':#test -> 50% valid, 50% test
		print "Now in :",source_data_path+traintest
		for song in os.listdir(source_data_path+traintest): 
			if os.path.isdir(source_data_path+traintest+'/'+song):
				song+='/'
				print "Now in :",source_data_path+traintest+'/'+song
				#mixture, sr = librosa.load(source_data_path+traintest+'/'+song+'mixture.wav') #default: sr=22050
				#vocals, sr = librosa.load(source_data_path+traintest+'/'+song+'vocals.wav')
				accompaniment, sr = librosa.load(source_data_path+traintest+'/'+song+'accompaniment.wav')
				#print mixture.shape,vocals.shape,accompaniment.shape
				#mixture_spec = np.abs(librosa.core.spectrum.stft(mixture, n_fft=2048))#default: hop_length=512, win_length=2048
				#vocals_spec = np.abs(librosa.core.spectrum.stft(vocals, n_fft=2048))
				accompaniment_spec = np.abs(librosa.core.spectrum.stft(accompaniment, n_fft=2048))
				#time_length = mixture_spec.shape[1]
                #time_length = vocals_spec.shape[1]
                time_length = accompaniment_spec.shape[1]
				#print mixture_spec.shape,vocals_spec.shape,accompaniment_spec.shape
				#mixture_spec = mixture_spec.T[:frame_wedth*(time_length/frame_wedth)]
				#vocals_spec = vocals_spec.T[:frame_wedth*(time_length/frame_wedth)
                accompaniment_spec = accompaniment_spec.T[:frame_wedth*(time_length/frame_wedth)]
				#print mixture_spec.shape,vocals_spec.shape,accompaniment_spec.shape

				#here use sparse split 
				#or with data augmentation?
				#mixture_spec = np.array(np.split(mixture_spec,time_length/frame_wedth))
				#vocals_spec = np.array(np.split(vocals_spec,time_length/frame_wedth))
                accompaniment_spec = np.array(np.split(accompaniment_spec,time_length/frame_wedth))
				#print mixture_spec.shape,vocals_spec.shape,accompaniment_spec.shape

				#mixture_data = np.concatenate((mixture_data,mixture_spec),axis=0)
				#vocals_data = np.concatenate((vocals_data,vocals_spec),axis=0)
                accompaniment_data = np.concatenate((accompaniment_data,accompaniment_spec),axis=0)
				#print mixture_data.shape,vocals_data.shape,accompaniment_data.shape
				#print mixture_data.shape
                #print vocals_data.shape
                print accompaniment_data.shape
				#break

        #np.save(training_data_path+traintest+'_mixture.npy',mixture_data[1:])
        #np.save(training_data_path+traintest+'_vocals.npy',vocals_data[1:])
        np.save(training_data_path+traintest+'_accompaniment.npy',accompaniment_data[1:])
        mixture_data = np.zeros([1,frame_wedth,freq_bin])
        vocals_data = np.zeros([1,frame_wedth,freq_bin])
        accompaniment_data = np.zeros([1,frame_wedth,freq_bin])




# trainX=[]
# trainY=[]
# validX=[]
# validY=[]
# testX=[]
# testY=[]

# trainX=np.array(trainX)
# trainY=np.array(trainY)
# validX=np.array(validX)
# validY=np.array(validY)
# testX=np.array(testX)
# testY=np.array(testY)

# include: 
# (human voice,human voice),(human voice+accompaniment,human voice), 
# (accompaniment, zero) as (x,y) (or maybe include other combinations)
# bass,drums,vocals,rest part in musdb data





