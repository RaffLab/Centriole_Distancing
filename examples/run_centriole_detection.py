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
    config = configparser.ConfigParser()

    # configfile = sys.argv[1]
    configfile = 'config_detection.txt'
    config = parse_config(configfile)

# =============================================================================
#   Check input and savefolder locations have been set.  
# =============================================================================
    if len(config['Experiment']['infolder']) == 0 or len(config['Experiment']['savefolder']) == 0:
        raise Exception('Input experiment folder location or Save experiment folder location have not been set!')
    
    print(config.sections())
    
    
# =============================================================================
#   Check input and savefolder locations have been set.  
# =============================================================================
    
    