#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:04:59 2019

@author: felix
"""

import numpy as np

def extract_dot_annotation_mask(img, c=[255,0,0]):

    r_mask = img[:,:,0]==c[0]
    g_mask = img[:,:,1]==c[1]
    b_mask = img[:,:,2]==c[2]

    mask = np.logical_and(r_mask, np.logical_and(g_mask, b_mask))
    
    return mask
    
    
def extract_dot_annotation_zstack(vidstack, c=[255,0,0]):
    
    dots = []

    for vid in vidstack:
        mask = extract_dot_annotation_mask(vid, c=c)
        dots.append(mask)
        
    return np.array(dots)
