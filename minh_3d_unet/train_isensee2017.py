#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:53:51 2017

@author: vhm
"""

from model import unet_model_3d

import numpy as np

from keras.utils import plot_model
from keras import callbacks
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

from data_handling import load_train_data, load_validatation_data

from unet3d.model import isensee2017_model
from model import dice_coef_loss
from unet3d.training import load_old_model, train_model

import configs

patch_size = configs.PATCH_SIZE
batch_size = configs.BATCH_SIZE

config = dict()
config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
config["image_shape"] = (256, 128, 256)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = (patch_size, patch_size, patch_size)  # switch to None to train on the whole image
config["nb_channels"] = 1
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))

config["n_labels"] = configs.NUM_CLASSES
config["n_base_filters"] = 16
config["all_modalities"] = ['t1']#]["t1", "t1Gd", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
config["deconvolution"] = False  # if False, will use upsampling instead of deconvolution
config["batch_size"] = batch_size
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 20  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.0001
config["depth"] = configs.DEPTH
config["learning_rate_drop"] = 0.5

image_type = '3d_patches'


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_gtruth_train = load_train_data()

    imgs_train = np.transpose(imgs_train, (0, 4, 1, 2, 3))
    imgs_gtruth_train = np.transpose(imgs_gtruth_train, (0, 4, 1, 2, 3))
    
    print('-'*30)
    print('Loading and preprocessing validation data...')
    print('-'*30)
    
    imgs_val, imgs_gtruth_val  = load_validatation_data()
    imgs_val = np.transpose(imgs_val, (0, 4, 1, 2, 3))
    imgs_gtruth_val = np.transpose(imgs_gtruth_val, (0, 4, 1, 2, 3))
    
    
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

   # create a model
    model = isensee2017_model(input_shape=config["input_shape"], n_labels=config["n_labels"],
                                          initial_learning_rate=config["initial_learning_rate"],
                                          n_base_filters=config["n_base_filters"],loss_function=dice_coef_loss)

    model.summary()



    
    #summarize layers
    #print(model.summary())
    # plot graph
    #plot_model(model, to_file='3d_unet.png')
    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    
    #============================================================================
    print('training starting..')
    log_filename = 'outputs/' + image_type +'_model_train.csv' 
    
    
    csv_log = callbacks.CSVLogger(log_filename, separator=',', append=True)
    
#    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')
    
    #checkpoint_filepath = 'outputs/' + image_type +"_best_weight_model_{epoch:03d}_{val_loss:.4f}.hdf5"
    checkpoint_filepath = 'outputs/' + 'weights.h5'
    
    checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    callbacks_list = [csv_log, checkpoint]
    callbacks_list.append(ReduceLROnPlateau(factor=config["learning_rate_drop"], patience=config["patience"],
                                           verbose=True))
    callbacks_list.append(EarlyStopping(verbose=True, patience=config["early_stop"]))

    #============================================================================
    hist = model.fit(imgs_train, imgs_gtruth_train, batch_size=config["batch_size"], nb_epoch=config["n_epochs"], verbose=1, validation_data=(imgs_val,imgs_gtruth_val), shuffle=True, callbacks=callbacks_list) #              validation_split=0.2,
        
     
    model_name = 'outputs/' + image_type + '_model_last'
    model.save(model_name)  # creates a HDF5 file 'my_model.h5'

    
if __name__ == '__main__':
    train_and_predict()