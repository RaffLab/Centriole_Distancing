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
    import pandas as pd 
    import numpy as np 

    import src.file_io as fio
    import src.image_fn as image_fn
    
# =============================================================================
#   Parse Config File  
# =============================================================================
    config = configparser.ConfigParser()

    # configfile = sys.argv[1]
    configfile = 'examples/config_export.txt'
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
    
# =============================================================================
#   Find the individual experiments.
# =============================================================================
    
    expts = fio.locate_experiment_series(infolder, key=config['Experiment']['expt_key'], exclude=['compiled'])

    # iterate over each experiment folder. 
    for expt in tqdm(expts[:]):

        dist_files = fio.locate_files(expt, key='.csv')
        # sort the files by numerical order
        dist_files, time = fio.natsort_files(dist_files, key='Image ', return_num=True)

        distance_tab = []
        intensity_tab = []

        # parse the attributes from the files.
        for ii, distfile in enumerate(dist_files[:]):
            
            distab = pd.read_csv(distfile)
            
            # flatten the distab. 
            dist_cols = np.hstack(['Dist %s' %(str(i).zfill(3)) for i in range(len(distab)+1)])
            dist_vals = np.hstack(['Time %s' %(str(int(ii))), distab['Distance[Pixels]'].values])
            SNR_vals = np.hstack(['Time %s' %(str(int(ii))), distab['SNR'].values])
            
            dist_row_entry = pd.DataFrame(dist_vals[None,:], columns=dist_cols)
            intensity_row_entry = pd.DataFrame(SNR_vals[None,:], columns=dist_cols)
            
            distance_tab.append(dist_row_entry)
            intensity_tab.append(intensity_row_entry)
            
        # concat both tables (now to make them lie on top of each other.)
        distance_tab = pd.concat(distance_tab, ignore_index=True, axis=0)
        intensity_tab = pd.concat(intensity_tab, ignore_index=True, axis=0)
        
#        # append new rows into intensity_tab
        header_tab = pd.DataFrame(index=np.arange(2), columns=intensity_tab.columns)
        header_tab.iloc[0] = np.nan
        header_tab.iloc[1] = np.hstack(['Intensity %s' %(str(i).zfill(3)) for i in range(len(intensity_tab.columns))])
        
        intensity_tab = pd.concat([header_tab,intensity_tab], ignore_index=True, axis=0)
        final_tab = pd.concat([distance_tab, intensity_tab], axis=0, ignore_index=True)

        _ , expt_name = os.path.split(expt)
        savefile_expt = os.path.join(savefolder, 'dist-SNR-template_'+expt_name+'.csv')
    
        final_tab.to_csv(savefile_expt, sep=',', index=None)

