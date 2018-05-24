import numpy as np
from CDAE2 import CDAE
from maskLSTM import LSTM
import librosa
import os 
import keras
import json
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
frame_wedth = 15
freq_bin = 1025
batch_size = 100
model = CDAE()
maskmodel = LSTM()
weight_file  =  './201804301130_CDAE.hdf5'  
if os.path.isfile(weight_file):
	print 'trained weight exists!'
	model.load_weights(weight_file)
	print 'complete load weights!!'
else:
	sys.exit()
mask_weight_file  =  'mask001.hdf5'  
if os.path.isfile(mask_weight_file):
	print 'mask trained weight exists!'
	maskmodel.load_weights(mask_weight_file)
	print 'complete load weights!!'
else:
	sys.exit()
#x_test= np.load('./data/test_mixture.npy')
# shape = list(x_test.shape) 
# shape.append(1)
# x_test = x_test.reshape(shape)
'''
y_test= np.load('./data/test_vocals.npy')
test_pad_len = batch_size - (len(x_test)%batch_size)
x_test = np.concatenate((x_test,np.zeros([test_pad_len,frame_wedth,freq_bin])),axis=0)
y_test = np.concatenate((y_test,np.zeros([test_pad_len,frame_wedth,freq_bin])),axis=0)

y_predict = model.predict(x_test,batch_size)
print x_test.shape,y_test.shape,y_predict.shape
print "mean abs difference:",np.abs(y_test-y_predict).mean()

np.save('LSTMpredict.npy',y_predict)
#'''

frame_wedth = 15
i = 0+1j
sr = 22050
#path = './musdb18/test/Zeno - Signs/mixture.wav'
path = './sodagreen.wav'
test_mix,sr = librosa.load(path)
test_spec = librosa.core.spectrum.stft(test_mix, n_fft=2048)

magnitude = np.abs(test_spec)
phase = np.angle(test_spec)
print "test_spec.shape,magnitude.shape:",test_spec.shape,magnitude.shape
time_length = magnitude.shape[1]
magnitude = magnitude.T[:frame_wedth*(time_length/frame_wedth)]
phase = phase.T[:frame_wedth*(time_length/frame_wedth)].T
x_test = np.array(np.split(magnitude,time_length/frame_wedth))
test_pad_len = batch_size - (len(x_test)%batch_size)
x_test = np.concatenate((x_test,np.zeros([test_pad_len,frame_wedth,freq_bin])),axis=0)
print "x_test.shape:",x_test.shape
# shape = list(x_test.shape) 
# shape.append(1)
# x_test = x_test.reshape(shape)
y_hat = model.predict(x_test,batch_size)
mask_y_hat = maskmodel.predict(x_test,batch_size)
print "y_hat.shape,mask_y_hat.shape:",y_hat.shape,mask_y_hat.shape
y_hat = y_hat*mask_y_hat

y_hat = np.reshape(y_hat,(-1,1025)).T
y_hat = y_hat[:,:phase.shape[1]]
print "y_hat.shape,phase.shape:",y_hat.shape,phase.shape
c = np.multiply( y_hat,np.exp(i*phase))
y_wav = librosa.istft(c)
librosa.output.write_wav('./sodagreen_test.wav',y_wav,sr)


