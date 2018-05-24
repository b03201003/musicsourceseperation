import os
import scipy.io.wavfile as wav
import numpy as np

# generate accompaniment, or some instrument mix voice

source_data_path = 'musdb18/'

for traintest in os.listdir(source_data_path):
	if os.path.isdir(source_data_path+traintest):
		traintest = traintest+'/'
		print "Now in :",source_data_path+traintest
		for song in os.listdir(source_data_path+traintest): 
			#if song[-4:]!='.mp4':
				#print "Now in :",source_data_path+traintest+song
				#print os.path.isdir(source_data_path+traintest+song+'')
			if os.path.isdir(source_data_path+traintest+song):
				song = song+'/'
				print "Now in :",source_data_path+traintest+song
				#sr,vocals = wav.read('vocals.wav')
				sr,bass = wav.read(source_data_path+traintest+song+'bass.wav')
				sr,drums = wav.read(source_data_path+traintest+song+'drums.wav')
				sr,other = wav.read(source_data_path+traintest+song+'other.wav')
				#sr,mixture = wav.read('miture.wav')
				accompaniment = np.add(bass,np.add(drums,other))
				if os.path.exists(source_data_path+traintest+song+'accompaniment.wav'):
					os.remove(source_data_path+traintest+song+'accompaniment.wav')
				wav.write(source_data_path+traintest+song+'accompaniment.wav',sr,accompaniment)

