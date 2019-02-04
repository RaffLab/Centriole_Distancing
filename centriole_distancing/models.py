#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module contains functions to aid in the building of the CNN architecture used by us in the paper

"""

import numpy as np
    
def distance_model(n_channels=1, shape=None, N_Cls=2):
    """ Builds the CNN U-net model described in paper in Keras to learn centriole distancing on local image patches.

    Parameters
    ----------
    n_channels : int
        number of input image channels.
    shape : None or 3-tuple
        (n_rows x n_cols x n_channel) assuming tensorflow format. If None, the model is fully convolutional and can be applied to any size input.
    N_Cls : int
        number of output channels, should equal the number of channels in annotations.
    
    Returns
    -------
    model : a Keras Model class
        CNN model instance. Use model.summary() to get a print out of architecture.

    """
    from keras.layers import Input, Conv2D, MaxPooling2D, merge

    if shape is None:
        ISZ = None
        inputs = Input((ISZ, ISZ, n_channels))
    else:
        inputs = Input(shape)

    # down-pooling branch
    conv1 = Conv2D(32, (3, 3), activation='selu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='selu', padding='same')(conv1)    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='selu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='selu', padding='same')(conv2)

    # up-pooling branch
    up1 = merge([UpSampling2D(size=(2, 2))(conv2), conv1], mode='concat', concat_axis=-1)
    conv3 = Conv2D(32, (3, 3), activation='selu', padding='same')(up1)
    conv3 = Conv2D(32, (3, 3), activation='selu', padding='same')(conv3)
    output = Conv2D(N_Cls, (1, 1), activation='relu')(conv3)

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model


def train_model(train_data, test_data, model, model_name, optimizer, loss='mse', scale_factor=1000., batch_size=128, max_epochs=200, early_stop=True, plot_history=True):
    """ Code to train a given model and save out to the designated path as given by 'model_name'

    Parameters
    ----------
    train_data : 2-tuple
        (train_x, train_y) where train_x is the images and train_y is the Gaussian dot annotation images in the train split.
    test_data : 2-tuple
        (test_x, test_y) where test_x is the images and test_y is the Gaussian dot annotation images in the test split.
    model : a Keras model
        a defined Keras model
    optimizer : Keras optimizer object
        the gradient descent optimizer e.g. Adam, SGD instance used to optimizer the model. We used Adam() with default settings.
    loss : string 
        one of 'mse' (mean squared error) or 'mae' (mean absolute error)
    scale_factor : None or float
        multiplicative factor to apply to annotation images to increase the gradient in the backpropagation
    batch_size : int
        number of images to batch together for training 
    max_epochs : int
        the maximum number of epochs to train for if early_stop is enabled else this is the number of epochs of training.
    early_stop : bool
        if True, monitors the minimum of the test loss. If loss does not continue to decrease for a set duration, stop the training and return the model with the best test loss.
    plot_hist : bool
        if True, plots the training and test loss over the training period on the same axes for visualisation.
        
    Returns
    -------
    None : void
        This function will simply save the model to the location given by model_name.

    """
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    import pylab as plt 
    
    train_x, train_y = train_data
    test_x, test_y = test_data

    if scale_factor is not None:
        train_y = train_y * float(scale_factor)
        test_y = test_y * float(scale_factor)

    # compile the model with chosen optimizer.
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) 
    
    if early_stop:
        """ Set some early stopping parameters """
        early_stop = EarlyStopping(monitor='val_loss', 
                                min_delta=0.001, 
                                patience=15, 
                                mode='min', 
                                verbose=1)
        
        checkpoint = ModelCheckpoint(model_name, 
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='min', 
                                    period=1)

        history = model.fit(train_x, train_y,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(test_x, test_y), shuffle=True, 
                callbacks = [early_stop, checkpoint])
    else:
        history = model.fit(train_x, train_y,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(test_x, test_y), shuffle=True)
        model.save(model_name) # save the whole model state.
    
    if plot_history:
        plt.figure()
        plt.plot(history.history['loss'], 'r', label='train loss')
        plt.plot(history.history['val_loss'], 'g', label='test loss')
        plt.legend()
        plt.show()

    return []


if __name__=="__main__":
    
    import numpy as np 
    import pylab as plt 
    from tqdm import tqdm
    from keras.models import Model
    from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Concatenate, BatchNormalization, Activation
    from keras.optimizers import Adam
    from scipy.misc import imsave 
    import os 
    import cv2
    from skimage.exposure import rescale_intensity
    from skimage.filters import sobel
    import scipy.io as spio 
    
    np.random.seed(1674)
    
#==============================================================================
#   Load the train-test_data
#==============================================================================
    
    in_files = ['Training_Testing_patches_sectioned-Early.mat',
                'Training_Testing_patches_sectioned-Mid.mat',
                'Training_Testing_patches_sectioned-Late.mat']

###==============================================================================
###   Distort samples .. 
###==============================================================================
    stage = 'Early'
#    # Load up the single model first. 
#    (X_train,Y_train,D_train), (X_test,Y_test,D_test) = compile_train_test_data(in_files, typ='single', which=0)
#    
#    X_train = X_train / 255.
#    
####==============================================================================
####   Apply Deformations
####==============================================================================
#    train_x, train_y = apply_elastic_transform_intensity_multi(X_train/255., Y_train, strength=0.2, shift=0, N=50) # can try to use strength lower. 
#    train_x, train_y_dots = create_dot_annotations_multi(train_x, train_y, sigma=2, thresh=0) # turn into dots. 
#    
#    test_x, test_y_dots = create_dot_annotations_multi(X_test/255., Y_test, sigma=2, thresh=0) # turn into dots. 
#    
##     do we deform also ? -> hm..... no?? 
##    X_test = X_test / 255.
##    test_x, test_y_dots = create_dot_annotations_multi(X_test, Y_test, sigma=2, thresh=0)
##    test_x = rescale_intensity_stack(test_x)
#    
#    
#    # try saving the data
#    spio.savemat('pooled_augmented_train-test-compress-%s.mat' %(stage), {'X_train':train_x,
#                                                     'Y_train_dots':train_y_dots, 
#                                                     'X_test':test_x,
#                                                     'Y_test_dots':test_y_dots}, do_compression=True)
    
    
    
    # too noisy for validaton -> augment the test data too !.
#    test_x, test_y = apply_elastic_transform_intensity_multi(X_test/255., Y_test, strength=0.2, shift=0, N=50) # can try to use strength lower. 
#    test_x, test_y_dots = create_dot_annotations_multi(test_x, test_y, sigma=2, thresh=0) # turn into dots. 
#    
#    
#    train_x = train_x[:,:,:,None]
#    train_y_dots = train_y_dots[:,:,:,None]
#    test_x = test_x[:,:,:,None]
#    test_y = test_y_dots[:,:,:,None]
    train_test_data = spio.loadmat('pooled_augmented_train-test-compress-%s.mat' %(stage))
    
    train_x = train_test_data['X_train']
    train_y_dots = train_test_data['Y_train_dots']
    test_x = train_test_data['X_test']
    test_y_dots = train_test_data['Y_test_dots']
    
    train_x = train_x[:,:,:,None]
    train_y_dots = train_y_dots[:,:,:,None]
    test_x = test_x[:,:,:,None]
    test_y = test_y_dots[:,:,:,None]
    
    #==============================================================================
    #    U-net Model Construction.  
    #==============================================================================
    # save out and construct a FCNN....(probably need a scale transform)
    ISZ = None
    N_Cls = 2
    
    inputs = Input((ISZ, ISZ, 1))
    conv1 = Conv2D(32, (3, 3), activation='selu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='selu', padding='same')(conv1)
    
#    # replace to get batch normalization 
#    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
#    conv1 = BatchNormalization()(conv1)
#    conv1 = Activation('selu')(conv1)
#    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
#    conv1 = BatchNormalization()(conv1)
#    conv1 = Activation('selu')(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='selu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='selu', padding='same')(conv2)
    
#    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
#    conv2 = BatchNormalization()(conv2)
#    conv2 = Activation('selu')(conv2)
#    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
#    conv2 = BatchNormalization()(conv2)
#    conv2 = Activation('selu')(conv2)
    
#    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
##    conv3 = Conv2D(128, (3, 3), activation='selu', padding='same')(pool2)
##    conv3 = Conv2D(128, (3, 3), activation='selu', padding='same')(conv3)
#    
##    conv3 = Conv2D(64, (3, 3), padding='same')(pool2)
##    conv3 = BatchNormalization()(conv3)
##    conv3 = Activation('relu')(conv3)
##    conv3 = Conv2D(64, (3, 3), padding='same')(conv3)
##    conv3 = BatchNormalization()(conv3)
##    conv3 = Activation('relu')(conv3)
#    
#    conv3 = Conv2D(128, (3, 3), activation='selu', padding='same')(pool2)
#    conv3 = Conv2D(128, (3, 3), activation='selu', padding='same')(conv3)
#    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
##
##    conv4 = Conv2D(72, (3, 3), activation='relu', padding='same')(pool3)
##    conv4 = Conv2D(72, (3, 3), activation='relu', padding='same')(conv4)
##    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
##    
##    conv5 = Conv2D(72, (3, 3), activation='relu', padding='same')(pool4)
##    conv5 = Conv2D(72, (3, 3), activation='relu', padding='same')(conv5)
##    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
###
##    conv6 = Conv2D(90, (3, 3), activation='relu', padding='same')(pool5)
##    conv6 = Conv2D(90, (3, 3), activation='relu', padding='same')(conv6)
##
##    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv5], mode='concat', concat_axis=-1)
##    conv7 = Conv2D(72, (3, 3), activation='relu', padding ='same')(up7)
##    conv7 = Conv2D(72, (3, 3), activation='relu', padding ='same')(conv7)
##
#    up8 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=-1)
#    up8 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv3), conv2])
#    conv8 = Conv2D(64, (3, 3), activation='selu', padding='same')(up8)
#    conv8 = Conv2D(64, (3, 3), activation='selu', padding='same')(conv8)
#
#    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv3], mode='concat', concat_axis=-1)
#    up9 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv2), conv1])
#    conv9 = Conv2D(32, (3, 3), activation='selu', padding='same')(up9)
#    conv9 = Conv2D(32, (3, 3), activation='selu', padding='same')(conv9)
#
    up10 = merge([UpSampling2D(size=(2, 2))(conv2), conv1], mode='concat', concat_axis=-1)
    conv10 = Conv2D(32, (3, 3), activation='selu', padding='same')(up10)
    conv10 = Conv2D(32, (3, 3), activation='selu', padding='same')(conv10)
#    
#    up11 = merge([UpSampling2D(size=(2, 2))(conv10), conv1], mode='concat', concat_axis=-1)
#    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(up11)
#    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv11)
#    conv10 = merge([conv9, inputs], mode='concat', concat_axis=-1)
#    up10 = merge([inputs, conv9], mode='concat', concat_axis=-1) # also feed in the original image as well as the upsampled. 
    conv12 = Conv2D(N_Cls, (1, 1), activation='relu')(conv10)

    model = Model(inputs=inputs, outputs=conv12)
    model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy']) # avoid over detection ?

    model.summary()

    
    #==============================================================================
    #   Model Fitting with Early Stopping.   
    #==============================================================================
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    

    train_y_dots = train_y_dots* 1000.
    test_y_dots = test_y_dots * 1000.
    
    batch_size = 128
    epochs = 200
    
#        """ Set some early stopping parameters """
    early_stop = EarlyStopping(monitor='val_loss', 
                               min_delta=0.001, 
                               patience=15, 
                               mode='min', 
                               verbose=1)
    
    checkpoint = ModelCheckpoint('Multi-S1_32x32_selu_all_sigma2_mse-%s-notestaug' %(stage), 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min', 
                                 period=1)

    # single model
#    history = model.fit(train_x, train_y_dots[:,:,:,0,0][:,:,:,None],
#              batch_size=batch_size,
#              epochs=epochs,
#              validation_data=(test_x, test_y_dots[:,:,:,0][:,:,:,None]), shuffle=True, 
#              callbacks = [early_stop, checkpoint])
    
    history = model.fit(train_x, train_y_dots[:,:,:,0],
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(test_x, test_y_dots), shuffle=True, 
              callbacks = [early_stop, checkpoint])
    
    plt.figure()
    plt.plot(history.history['loss'], 'r')
    plt.plot(history.history['val_loss'], 'g')
#        
#####        
######### =============================================================================
#########   test
######### =============================================================================
#######
###    
    for i in range(len(test_x))[:10]:
        
#        imm = X_test[i]/255.
        imm = np.squeeze(test_x[i])
        imm = rescale_intensity(imm)
#        imm = equalize_adapthist((imm/255.)[:,:,0], clip_limit=0.01)[:,:,None]
#        imm = imm /255.
        
        out = model.predict(imm[None,:,:,None])/1000.
        
#        plt.figure()
#        plt.subplot(121)
#        plt.imshow(np.squeeze(imm), cmap='gray')
#        plt.subplot(122)
#        plt.imshow(np.squeeze(out))
#        plt.show()
        
        plt.figure()
        plt.subplot(131)
        plt.imshow(np.squeeze(imm), cmap='gray')
        plt.subplot(132)
        plt.imshow(np.squeeze(out[:,:,:,0]))
        plt.subplot(133)
        plt.imshow(np.squeeze(out[:,:,:,1]))
#        plt.subplot(154)
#        plt.imshow(np.squeeze(out))
#        plt.subplot(155)
#        plt.imshow(Y_test[i][:,:,1])
#        plt.show()
        
        print np.sum(np.squeeze(out))
#    
#    
#    
#    
    

        
        
        
        
        
    
    