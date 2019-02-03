#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This example is a simple pipeline command script to take an input configuration file and output the results.

"""
def parse_config(configfile):

    import configparser
    config = configparser.ConfigParser()
    
    f = open(configfile, 'r') # open as an iterable. 
    config.read_file(f)
    print('read configuration file')

    return config

if __name__=="__main__":

    import sys
    import configparser
    import pylab as plt 
    import os
    from tqdm import tqdm

    import src.file_io as fio
    import src.image_fn as image_fn
    
# =============================================================================
#   Parse Config File  
# =============================================================================
    config = configparser.ConfigParser()

    # configfile = sys.argv[1]
    configfile = 'examples/config_detection.txt'
    config = parse_config(configfile)

# =============================================================================
#   Check input and savefolder locations have been set.  
# =============================================================================
    if len(config['Experiment']['infolder']) == 0 or len(config['Experiment']['savefolder']) == 0:
        raise Exception('Input experiment folder location or Save experiment folder location have not been set!')
    
    
# =============================================================================
#   Check input and savefolder locations have been set.  
# =============================================================================
    infolder = config['Experiment']['infolder']
    savefolder = config['Experiment']['savefolder']; fio.mkdir(savefolder)
    
    tiffiles = fio.locate_files(infolder, key='.tif')
    
    print('Found %d files' %(len(tiffiles)))
    
    # Parse the settings 
    detect_ = config['Centriole_Detection']
    bg_ = config['Background_Filter']
 
    aniso_ = config['Anisotropic_Image_Filter']
    aniso_params = {'delta': float(aniso_['delta']), 
                    'kappa': float(aniso_['kappa']),
                    'iterations': int(aniso_['iterations'])}
    

    tslice = int(config['Input']['tslice'])
    is_img_slice = config['Input']['is_img_slice'] == 'True'
    filter_border = detect_['filter_border'] == 'True'
    filter_high_intensity_bg = detect_['filter_high_intensity_bg'] == 'True'
    filter_CV = detect_['filter_CV'] == 'True'
    remove_duplicates = detect_['remove_duplicates'] == 'True'
    debug = detect_['debug_detections'] == 'True'
# =============================================================================
#   Run the detection algorithm
# =============================================================================
    
    for ii in tqdm(range(len(tiffiles[:]))):
        
        filename = tiffiles[ii]
        if is_img_slice == False:
            if tslice == -1:
                img = fio.read_stack_img(filename)
            else:
                img = fio.read_stack_time_img(filename)
        
        detections = image_fn.detect_centrioles_in_img(img, 
                                                       int(detect_['centriole_size']), 
                                                       aniso_params, 
                                                       int(detect_['patch_size']), 
                                                       CV_thresh=float(detect_['CV_thresh']), 
                                                       tslice=tslice, 
                                                       is_img_slice=is_img_slice, 
                                                       filter_border=filter_border, 
                                                       filter_high_intensity_bg=filter_high_intensity_bg, 
                                                       remove_duplicates=remove_duplicates, 
                                                       filter_CV=filter_CV, 
                                                       separation= float(detect_['centriole_separation']), 
                                                       invert=False, 
                                                       minmass=float(detect_['centriole_minmass']), 
                                                       minoverlap=float(detect_['minoverlap']), 
                                                       bg_min_I=float(bg_['bg_min_i']), 
                                                       bg_max_area=int(bg_['bg_max_area']), 
                                                       bg_dilation=int(bg_['bg_dilation']), 
                                                       bg_invalid_check=float(bg_['bg_invalid_check']), 
                                                       debug=debug)
        
    # =============================================================================
    #   Save the outputs into .mat
    # =============================================================================
        savefile = filename.replace(infolder, savefolder)
        file_savefolder, savename = os.path.split(savefile); fio.mkdir(file_savefolder)
        savename = savename.replace('.tif', '-detections.pkl')
        
        # pickle the object, more flexible. 
        fio.save_obj(detections, os.path.join(file_savefolder, savename))
    