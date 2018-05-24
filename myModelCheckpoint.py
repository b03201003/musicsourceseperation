# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:13:29 2016

@author: choh
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import warnings
import sys
import os 
from keras import backend as K
from keras.callbacks import Callback

class myModelCheckpoint(Callback):
    '''Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then multiple files will be save with the epoch number and
    the validation loss.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the validation loss will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minization of the monitored. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
    '''
    def __init__(self, filepath, monitor='val_loss', verbose=1,
                 save_best_only=False, mode='min'):

        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.train_best = np.Inf
        self.reduce_rate  = 0.1
        self.epochs = 0 
        self.epochs_reduce = 30

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        self.epochs+=1
        filepath = self.filepath.format(epoch=epoch, **logs)

        loss = logs.get('loss')
        print('Now loss is ',loss)
        if loss<self.train_best:
            self.train_best = loss
        
        myPath = (os.getcwd()) #'./'
        fAll = open(myPath+'/myHistoryRecordAll.txt', 'a+')

        Record_text = fAll.read().split("\n")[:-1]
        #print('last part of record:',Record_text)
        valid_loss_record =[]
        for i,r in enumerate(reversed(Record_text)):
            if  "valid_loss" in Record_text[-(i+1)]:
                valid_loss_record.append(float(Record_text[-(i+1)].replace('valid_loss\t','')))
            if len(valid_loss_record)>=5:
                break
        print('valid loss record:',valid_loss_record)
        print('current loss average:',float(np.average(valid_loss_record)))
        lr = K.get_value(self.model.optimizer.lr)
        if  valid_loss_record[0] > float(np.average(valid_loss_record)) and self.epochs>self.epochs_reduce :
            #valid_loss_record[0] is the last valid_loss in Record_text
            self.epochs_reduce = self.epochs+100
            print('reduce learning rate')
            K.set_value(self.model.optimizer.lr,lr*self.reduce_rate )
        print('Now learning rate:',lr)
        task_str = filepath
#        task_num = str(self.best)
        task_num = str(logs.get(self.monitor))
        temp_txt = 'valid_loss'+'\t'+task_num+'\n'
        temp_txt2 = 'train_loss'+'\t'+str(loss)+'\n'
        print('self.epochs:',self.epochs)
        if epoch==0:
            temp_txt = 'epoch : 1'+'\t'+filepath+'\n'+temp_txt
        fAll.write(temp_txt)  
        fAll.write(temp_txt2)                   
        fAll.close()        
        
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                print('Now best value is ',self.best)
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    self.model.save_weights(filepath, overwrite=True)

                    # update only best, choh
                    fo = open(myPath+'/myHistory.txt', 'r+')
                    #fo = open('/home/choh/W_Python/myISMIR2016/Ver2.2/myHistory.txt','r+')
                    # 0.671092951992 
                    # 0.676543210992
                    whole_text = fo.read()
                    fo.seek(0, 0)
                    
                    task_str = filepath
                    task_num = str(self.best)
                    new_text=""
                    myPos =0
                    
                    for line in fo:
                        prev_pos = myPos
                        myPos = prev_pos+len(line) 
                        
                        line_pos = line.find(task_str)
                        if line_pos != -1 :                            
                            fo.seek(myPos-15, 0)
                            temp_str = fo.read(14)        
                            fo.seek(prev_pos, 0)
                            temp_str2 = fo.read(myPos-prev_pos)
                            
                            aa = temp_str2.strip(task_str)
                            bb = aa.strip("\n")
                            cc = bb.strip("\t")
                            
                            temp_num_length = len(cc)
                            temp_num_pos = line.find(cc)   
                            temp_num = float(cc)
                            
                            if temp_num<task_num:
                                task_num_str = str(task_num)
                                new_text = whole_text[:prev_pos+temp_num_pos]+task_num_str+whole_text[prev_pos+temp_num_pos+temp_num_length:]
                                break                        
                    fo.close()                    
                    
                    if len(new_text)!=0:
                        f = open(myPath+'/myHistory.txt', 'w+')
                        #f = open('/home/choh/W_Python/myISMIR2016/Ver2.2/myHistory.txt','w+')
                        #print 'update!'
                        f.write(new_text)
                        f.close()
                    else:
                        f = open(myPath+'/myHistory.txt', 'a') 
                        #f = open('/home/choh/W_Python/myISMIR2016/Ver2.2/myHistory.txt','a')                     
                        temp_txt = filepath+'\t'+task_num+'\n'
                        f.write(temp_txt)
                        f.close()
                        
                        
                        
                    
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            self.model.save_weights(filepath, overwrite=True)


            