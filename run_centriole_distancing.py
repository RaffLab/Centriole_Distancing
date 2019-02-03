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
    from keras.models import load_model
    from skimage.exposure import rescale_intensity
    from skimage.feature import peak_local_max
    import numpy as np 
    import pandas as pd

    import src.file_io as fio
    import src.image_fn as image_fn
    import src.distancing as distancing
    
# =============================================================================
#   Parse Config File  
# =============================================================================
    config = configparser.ConfigParser()

    # configfile = sys.argv[1]
    configfile = 'examples/config_distancing.txt'
    config = parse_config(configfile)

# =============================================================================
#   Check input and savefolder locations have been set.  
# =============================================================================
    if len(config['Experiment']['infolder']) == 0 or len(config['Experiment']['savefolder']) == 0:
        raise Exception('Input experiment folder location or Save experiment folder location have not been set!')
    
    
# =============================================================================
#   Load files and set config parameters. 
# =============================================================================
    infolder = config['Experiment']['infolder']
    savefolder = config['Experiment']['savefolder']; fio.mkdir(savefolder)
    
    # Parse the CNN settings 
    CNN_model_ = config['CNN']
    single_model = CNN_model_['single_model']
    CNN_multiple = float(CNN_model_['multiplier'])

    if single_model == 'True':
        cnn_model = load_model(CNN_model_['CNN_model'])
    else:
        cnn_model_early = load_model(CNN_model_['CNN_model_early'])
        cnn_model_mid = load_model(CNN_model_['CNN_model_mid'])
        cnn_model_late = load_model(CNN_model_['CNN_model_late'])

    # Parse the distancing settings
    CNN_distancing_ = config['Centriole_Distancing']
    min_I_CNN = CNN_distancing_['min_I_CNN'] == 'None'
    if min_I_CNN == True:
        min_I_CNN = None
    else:
        min_I_CNN = float(CNN_distancing_['min_I_CNN'])

    filt_I_CNN = CNN_distancing_['filt_I_CNN'] == 'True'
    min_distance_CNN = int(CNN_distancing_['min_distance_CNN'])
    dist_thresh_CNN = float(CNN_distancing_['dist_thresh_CNN'])
    ratio_thresh_CNN = float(CNN_distancing_['ratio_thresh_CNN'])
    dist_thresh_GMM = float(CNN_distancing_['dist_thresh_GMM'])
    n_samples_GMM = int(CNN_distancing_['n_samples_GMM'])
    debug_CNN = CNN_distancing_['debug_viz'] == 'True'

    # if single model we don't care and pool everything else, we have to sort numericall the files to get early, mid, late.
    if single_model == 'True':
        pklfiles = fio.locate_files(infolder, key='detections.pkl') #looking for a particular signature. 
        print('Found %d centriole detection files' %(len(pklfiles)))


        # do distancing 


    else:
        expts = fio.locate_experiment_series(infolder, key=config['Experiment']['expt_key'])
        
        for expt in expts[:1]:
            pklfiles = fio.locate_files(expt, key='detections.pkl')
            pklfiles = fio.natsort_files(pklfiles, splitkey='_')

            n_files = len(pklfiles)

            for i in range(n_files)[:1]:
                filename = pklfiles[i]  
                basename = os.path.split(filename)[-1]
                saveimgfolder = filename.replace('.pkl', '') # same folder as detections.
                fio.mkdir(saveimgfolder)

                # load detection object and load the cropped detections. 
                detect_obj = fio.load_obj(filename)
                centriole_patches = detect_obj['cropped_detections'] # this is not yet projected. 
                max_slice_im = detect_obj['max_proj_full_img']
                max_slice_bg = detect_obj['background_mask']
                centriole_centroids = detect_obj['centriole_centroids']

                n_centrioles = len(centriole_patches)

                # preprocess (rescaling intensities)
                if len(centriole_patches.shape) == 4:
                    centriole_patches, best_slice_indices = image_fn.find_best_focus_stacks(centriole_patches)
                centriole_patch_input = np.concatenate([rescale_intensity(p*1.)[None,:] for  p in centriole_patches], axis=0)

                patch_size = centriole_patch_input.shape[1] # assumed square

                # pass through CNN to get predictions 
                if i < int(config['Experiment']['early']): 
                    cnn_out = np.squeeze(cnn_model_early.predict(centriole_patch_input[:,:,:,None])/CNN_multiple)
                elif i >= n_files + int(config['Experiment']['late']): 
                    cnn_out = np.squeeze(cnn_model_late.predict(centriole_patch_input[:,:,:,None])/CNN_multiple)
                else:
                    cnn_out = np.squeeze(cnn_model_mid.predict(centriole_patch_input[:,:,:,None])/CNN_multiple)
                
                """
                Aggregate the statistics. 
                """
                centriole_props = []
                centriole_dists = []
                peak_ids = []
                peak_intensities = []
                bg_peak_intensities = []

                # interpreting the CNN peaks.
                for ii in range(n_centrioles):

                    cnn_pred_centriole = cnn_out[ii,:,:,0]
                    cnn_pred_centre = cnn_out[ii,:,:,1]

                    # 1. locate centriole peaks. (there might be none? depending on the quality of initial detection?)
                    n_cnn_peaks_raw, cnn_peaks = distancing.detect_2d_peaks(cnn_pred_centriole, centriole_patch_input[ii], min_distance=min_distance_CNN, filt=filt_I_CNN, thresh=min_I_CNN)

                    if len(cnn_peaks) > 0:

                        # 2. locate centres between the points. 
                        cnn_centre = peak_local_max(cnn_pred_centre, num_peaks=1) # only the highest peak.

                        if len(cnn_centre) == 0: 
                            cnn_centre = np.array([[centriole_patch_input[ii].shape[0]//2, centriole_patch_input[ii].shape[1]//2]]) 
                        
                        # 3. associate centres to filter centriole locations. # how many to associate? Different strategy. 
                        cnn_peaks_filt = distancing.associate_peaks2centre_single(cnn_peaks, cnn_centre, dist_thresh=dist_thresh_CNN, ratio_thresh=ratio_thresh_CNN)
                        
                        # two cases: (if only one then run the GMM on the CNN.)
                        if len(cnn_peaks_filt) == 1: 
                            cnn_peaks_filt = distancing.fitGMM_patch_post_process( cnn_pred_centriole, n_samples=n_samples_GMM, max_dist_thresh=dist_thresh_GMM)
                            
                        peak_dist = np.linalg.norm(cnn_peaks_filt[0] - cnn_peaks_filt[1]) # find the distance between the two peaks. 
                        centriole_props.append(cnn_peaks_filt)
                        centriole_dists.append(peak_dist)
                        peak_ids.append(ii)

                        # 4. compute the associated intensity SNR associated with the patch.
                        patch_mean_I = np.mean(centriole_patches[ii])
                        patch_bg_I = np.mean(max_slice_im[max_slice_bg==1])

                        peak_intensities.append(patch_mean_I)
                        bg_peak_intensities.append(patch_bg_I)

                        """
                        (Optional) Visualisation and debugging of the peak detections
                        """
                        if debug_CNN:
                            # writes out a visualisation in the same processing folder. 
                            fig, ax = plt.subplots(nrows=1, ncols=3)
                            ax[0].imshow(centriole_patches[ii], cmap='gray')
                            ax[0].plot(cnn_centre[:,1], cnn_centre[:,0], 'b.')
                            ax[0].plot(cnn_peaks_filt[:,1], cnn_peaks_filt[:,0], 'ro')
                            
                            for jj in range(len(cnn_peaks_filt)):
                                ax[0].text(cnn_peaks_filt[jj,1]+1, cnn_peaks_filt[jj,0]+1, str(jj+1), fontsize=10, color='r', weight='bold')
                            
                            ax[1].imshow(cnn_pred_centriole, cmap='coolwarm')
                            ax[1].plot(cnn_peaks[:,1], cnn_peaks[:,0], 'go') # plots all possible 
                            
                            ax[2].imshow(cnn_pred_centre, cmap='coolwarm')
                            ax[2].plot(cnn_centre[:,1], cnn_centre[:,0], 'bo')
                            
                            ax[0].axis('off')
                            ax[1].axis('off')
                            ax[2].axis('off')

                            """
                            Saving the respective patch file. 
                            """
                            fig.savefig(os.path.join(saveimgfolder, str(ii+1).zfill(4)+'.svg'), dpi=300, bbox_inches='tight')
                            plt.close()

                """
                Compile the statistics for saving
                """
                centriole_dists = np.hstack(centriole_dists)
                peak_ids = np.hstack(peak_ids)
                peak_intensities = np.hstack(peak_intensities) 
                bg_peak_intensities = np.hstack(bg_peak_intensities)

                all_centriole_ids_txt = np.hstack([str(tt+1).zfill(4) for tt in range(len(peak_ids))]) # rename peak ids.
                
                centriole_stats = np.vstack([all_centriole_ids_txt, centriole_dists, peak_intensities, bg_peak_intensities, peak_intensities/bg_peak_intensities.astype(np.float)]).T
                tab_out = pd.DataFrame( centriole_stats,
                                        index=None, 
                                        columns = ['ID', 'Distance[Pixels]', 
                                                    'mean_intensity_both',  
                                                    'mean_background_intensity', 
                                                    'SNR'])
                tab_out.to_csv(os.path.join(saveimgfolder, 'distances_intensity_ML_'+basename.replace('.pkl', '.csv')), index=None)
                
           
                """
                if debug_CNN: save the final detections with identified centriole locations on the max. projected slice image. 
                """
                ##==============================================================================
                ##   Global plotting of results. 
                ##==============================================================================
                nrows, ncols = max_slice_im.shape
                height = float(nrows)
                width = float(ncols)
                
                fig = plt.figure()
                fig.set_size_inches(width/height, 1, forward=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(max_slice_im, cmap='gray')
        
                for i in range(len(peak_ids))[:]:
                    
                    patch_centre_coord = centriole_centroids[peak_ids[i]] 
                    local_patch_coord_adjust = centriole_props[i] - np.array([patch_size/2., patch_size/2.]).ravel()[None,:]
                    local_mean_patch_adjust = np.mean(local_patch_coord_adjust, axis=0)
                    glob_cent_centroid = patch_centre_coord + local_mean_patch_adjust
                    
                    """
                    Draw the square
                    """
                    x1_ = glob_cent_centroid[1] - patch_size
                    y1_ = glob_cent_centroid[0] - patch_size
                    x2_ = x1_ + 2*patch_size
                    y2_ = y1_ + 2*patch_size
                    
                    plt.text(x1_-1, y1_-2, str(i+1).zfill(3), fontsize=1.2, color='w')
                    plt.plot([x1_, x2_, x2_, x1_, x1_ ], [y1_, y1_, y2_, y2_, y1_], lw=0.2, color='w')

                plt.xlim([0, ncols])
                plt.ylim([nrows, 0])
                
                """
                saving out the global file too. 
                """
                plt.savefig(os.path.join(saveimgfolder, basename.replace('.pkl', '_full-image-cnn-detections.svg')), dpi = height) 
                plt.close()

                """
                save out the global mask. 
                """
                fig = plt.figure()
                fig.set_size_inches(width/height, 1, forward=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(max_slice_bg)
                plt.xlim([0, ncols])
                plt.ylim([nrows, 0])
                plt.savefig(os.path.join(saveimgfolder, basename.replace('.pkl', '_full-image-mask.svg')), dpi = height) 
                plt.close()



                

