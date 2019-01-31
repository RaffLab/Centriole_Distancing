#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:04:59 2019

@author: felix
"""

def read_multiimg_PIL(tiffile):
    
    """
    Use pillow library to read .tif/.TIF files. (single frame)
    
    Input:
    ------
    tiffile: input .tif file to read, can be multipage .tif (string)
    frame: desired frarme number given as C-style 0-indexing (int)

    Output:
    -------
    a numpy array that is either:
        (n_frames x n_rows x n_cols) for grayscale or 
        (n_frames x n_rows x n_cols x 3) for RGB

    """

    from PIL import Image
    import numpy as np
    import pylab as plt 
    
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

    return np.concatenate(imgs, axis=0)


def locate_centriole_files(infolder, key='.tif', exclude=None):
    
    """
    Locate the stack files of centriole images.
    
    Input:
    ------
    infolder: top-level folder of where the .tifs can be found
    key: the extension of the file
    
    Output:
    -------
        files: list of sorted filepaths.
    """
    
    import os
    import numpy as np
    
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