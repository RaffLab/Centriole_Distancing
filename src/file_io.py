#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module contains scripts for file input/output manipulation.

"""

import numpy as np 
import image_fn

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
