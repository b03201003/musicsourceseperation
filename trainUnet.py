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


model = Unet(input_shape=(512,128,1))
weight_file  =  './Unet.hdf5'   
if os.path.isfile(weight_file):
	print 'trained weight exists!'
	model.load_weights(weight_file)
	print 'complete load weights!!'

batch_size = 32
if sys.argv[1]== 'train':
	train_input_spec,train_output_spec = preprocess(train_paths)
	evaluate_input_spec,evaluate_output_spec = preprocess(test_paths)
	TrainEpochs = int(sys.argv[3])
	for i in range(TrainEpochs):
		History = model.fit_generator(image_generator(train_input_spec,train_output_spec),steps_per_epoch=len(train_input_spec)/batch_size ,epochs=1)
		#History = CrossModalityCNNmodel.fit(x=input_list,y=label,batch_size=batch_size,epochs=1,validation_split=0.2)
	#CrossModalityCNNmodel.save('myModel.h5')
		model.save_weights('./Unet.hdf5', overwrite=True)
		loss = model.evaluate_generator(image_generator(evaluate_input_spec,evaluate_output_spec),steps=len(evaluate_input_spec)/batch_size)
		print "{}-th epoch testing loss:{}\n".format(i,loss)
elif sys.argv[1]== 'predict':
	#predict_path = test_paths[0]
	predict_path = test_paths[len(test_paths)/2+1:len(test_paths)/2+2]
	print "predict_path:",predict_path
	#predict_data,predict_label = Get_nib_dataNlabel(predict_path)
	#predict_data =  predict_data.reshape((4,-1,240,240,1))
	#print predict_data.shape,predict_label.shape
	#predict_inputs = [modality for modality in predict_data]
	predict_outputs = model.predict_generator(image_generator(predict_path),steps = 5)
	print "predict_outputs.shape:",predict_outputs.shape
	np.save('Unetprediction.npy',predict_outputs)