#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module contains scripts for file input/output manipulation.

"""

import numpy as np 
import image_fn

def parse_config(configfile):

    import configparser
    config = configparser.ConfigParser()
    
    f = open(configfile, 'r') # open as an iterable. 
    config.read_file(f)
    print('read configuration file')

    return config

def save_obj(obj, savepath ):
    import pickle
    with open(savepath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(loadpath ):
    import pickle
    with open(loadpath, 'rb') as f:
        return pickle.load(f)

def read_rgb(imgfile):
    """Reads RGB image file

    Parameters
    ----------
    imgfile : string
        input file location.
    
    Returns
    -------
    img : numpy array
        An image where the channels are stored in the third dimension, such that
            (n_rows x n_cols): a gray-image.
            (n_rows x n_cols x 3): an RGB-image. 
            (n_rows x n_cols x 4): an RGBA-image.
        
    """
    from skimage.io import imread
    
    img = imread(imgfile)
    
    return img

def read_multiimg_PIL(tiffile):
    """Reads multipage .tif file

    note: z_slice data (bioformats) is flattened and coerced into the format for grayscale above i.e. 'n_frames' = n_timepoints x n_slices

    Parameters
    ----------
    tiffile : string
        input .tif file location.
    
    Returns
    -------
    imgs : numpy array
        (n_frames x n_rows x n_cols) for grayscale, or
        (n_frames x n_rows x n_cols x 3) for RGB
        
    """
    from PIL import Image
    
    img = Image.open(tiffile)

    imgs = []
    read = True

    frame = 0

    while read:
        try:
            img.seek(frame) # select this as the image
            imgs.append(np.array(img)[None,:,:])
            
            frame += 1
        except EOFError:
            # Not enough frames in img
            break

    imgs = np.concatenate(imgs, axis=0)

    return imgs

def read_stack_img(tiffile):
    """Utility function to read an (n_frames x n_rows x n_cols) grayscale/RGB image stack, converting input into uint8.

    note: n_frames is assumed to be either #. of time slices or #. of z-slices.

    Parameters
    ----------
    tiffile : string
        input .tif file location.
    
    Returns
    -------
    zstack_img : numpy array
        (n_frames x n_rows x n_cols) for grayscale, or
        (n_frames x n_rows x n_cols x 3) for RGB
        
    """
    zstack_img = read_multiimg_PIL(tiffile) 
    zstack_img = image_fn.uint16_2_uint8(zstack_img)
    
    return zstack_img

def read_stack_time_img(tiffile, n_timepoints, n_slices):
    
    """Utility function to read an (n_timepoints x n_slices x n_rows x n_cols) grayscale/RGB image stack, converting input into uint8.

    Parameters
    ----------
    tiffile : string
        input .tif file location.
    n_timepoints : int
        number of expected timepoints in the image.
    n_slices : int
        number of expected z-slices.
    
    Returns
    -------
    zstack_img : numpy array
        (n_timepoints x n_slices x n_rows x n_cols) for grayscale, or
        (n_timepoints x n_slices x n_rows x n_cols x 3) for RGB

    """
    zstack_img = read_stack_img(tiffile) 
    
    if len(zstack_img.shape) == 3:
        _, nrows, ncols = zstack_img.shape
        zstack_img = np.reshape(zstack_img, (n_timepoints, n_slices, zstack_img.shape[1], zstack_img.shape[2]))        
    
    elif len(zstack_img.shape) == 4:
        _, nrows, ncols, _ = zstack_img.shape
        zstack_img = np.reshape(zstack_img, (n_timepoints, n_slices, zstack_img.shape[1], zstack_img.shape[2], zstack_img.shape[3]))        
    
    return zstack_img


def locate_files(infolder, key='.tif', exclude=None):
    """Locate files given by a certain extension given by the 'key' parameter with optional keyword exclusion using 'exclude'.
    
    Parameters
    ----------
    infolder : string
        top-level folder location of where the files can be found. The function will walk the entire subdirectories underneath.
    key : string 
        the extension of the files being searched for e.g. '.csv', '.tif'
    exclude : list of strings (default=None)
        keywords within files one wishes to be excluded.

    Returns
    -------
    files : numpy array
        an array of sorted filepaths.
    """
    
    import os

    files = []
    
    for root, dirs, files_ in os.walk(infolder):
        for f in files_:
            if key in f and '._' not in f:
                if exclude is not None:
                    val = 0
                    for ex in exclude:
                        val+=int(ex in root)
                    if val == 0:
                        path = os.path.join(root, f)
                        files.append(path)
                else:
                    path = os.path.join(root, f)
                    files.append(path)
                
    files = np.sort(files)
                
    return files


def locate_experiment_series(infolder, key='Series'):
    """Locate top-level experiment folders given a keyword.
    
    Parameters
    ----------
    infolder : string
        top-level folder location of where the files can be found. The function will walk the entire subdirectories underneath.
    key : string 
        the keyword common to experiments. 

    Returns
    -------
    dirs_ : numpy array
        an array of sorted folderpaths.
    """
    import os 
    dirs_ = []
    
    for root, dirs, files in os.walk(infolder):
        for d in dirs:
            if key in d:
                dirs_.append(os.path.join(root, d))
                
    return np.sort(dirs_)
    
    
def natsort_files( files, splitkey='_'):
    """Sort the detected files in numerical order of acquisition based on integer naming. 
    
    Note: this function is not very generic at present. use regex instead. 

    Parameters
    ----------
    files : string
        filepaths
    key : string 
        the keyword common to experiments. 

    Returns
    -------
    dirs_ : numpy array
        an array of sorted folderpaths.
    """
    import os 
    nums = np.hstack([int((os.path.split(f)[1]).split(splitkey)[0].split()[1]) for f in files])
    sort_order = np.argsort(nums)
    
    return files[sort_order]


def mkdir(directory):
    """Checks if given directory path exists, if not creates it.
    
    Parameters
    ----------
    directory : string
        folderpath location. (Does not have to exist)

    Returns
    -------
    None

    """
    import os 

    if not os.path.exists(directory):
        os.makedirs(directory)
        
    return []


def save_multipage_tiff(np_array, savename):
    """Writes out a numpy array of images out as a multipage tiff file.
    
    Parameters
    ----------
    np_array : numpy array
        an (n_frames x n_rows x n_cols x n_channels) numpy array 
    savename : string 
        filepath to save to
    
    Returns
    -------
    None 

    """
    from tifffile import imsave
    
    imsave(savename, np_array.astype(np.uint8))
    
    return [] 
