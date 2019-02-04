#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module contains scripts to validate detections using mAP and precision-recall curves.

"""
import numpy as np 

def voc_ap(rec, prec):
    """ This function adapts the official Matlab code of VOC2012 to compute precision-recall curve.
  
    Code for this part is adapted from https://github.com/Cartucho/mAP. 

    Parameters
    ----------
    rec : numpy array or list 
        the recall at each threshold level.
    prec : numpy array or list
        the precision at each threshold level
        
    Returns
    -------
    ap : float
        the average precision
    mrec : list
        the recall which the average precision was computed with.
    mpre :list
        the precision which is forced to be monotonically decreasing for evaluating average precision.

    """
    rec = list(rec)
    prec = list(prec)

    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
    This part makes the precision monotonically decreasing (goes from the end to the beginning)
    """
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
    This part creates a list of indexes where the recall changes
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
    The Average Precision (AP) is the area under the curve (numerical integration)
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    
    return ap, mrec, mpre


def eval_AP_detections(all_gt_detections, all_pred_detections, dist_threshold=3):
    """ This function evaluates AP of (x,y) coordinate detections between manual annotations and predicted detections
  
    Parameters
    ----------
    all_gt_detections : numpy array 
        an (N x 3) array with N rows, each row of form (image_id, x, y) or (image_id, y, x)
    all_pred_detections : numpy array
        an (N x 4) array with N rows, each row of form (image_id, x, y, confidence) or (image_id, y, x, confidence) where confidence is a score of the goodness of detection e.g. CV for centriole pairs or CNN output intensity for individual centrioles
    dist_threshold : float
        Maximum distance (in pixels) below which we designate a positive match.
        
    Returns
    -------
    ap : float
        the average precision
    recall : list
        the recall curve for each threshold
    precision :list
        the precision curve for each threshold

    """    
    # initialise counters.
    fp = np.zeros(len(all_gt_detections)) # how many to get wrong, 
    tp = np.zeros(len(all_pred_detections)) # how many we get right 

    match_count_GT = np.zeros(len(all_gt_detections))
    
    # rank the all_pred_detections. 
    all_pred_detections_sort = all_pred_detections[all_pred_detections[:,-1].argsort()[::-1]] # sort by confidence, high -> low
        
    for ii in range(len(all_pred_detections))[:]:
        peak = all_pred_detections_sort[ii]

        select_relevant = np.arange(len(all_gt_detections))[all_gt_detections[:,0] == peak[0]] # fetch the right image id.
        relevant_gt_peaks = all_gt_detections[select_relevant] # which ones we consider. 
        relevant_match_count_GT = match_count_GT[select_relevant]
        
        # attempt to match. 
        if len(relevant_gt_peaks) == 0:
            fp[ii] = 1 # false peak. 
        else:
            peak_dist = relevant_gt_peaks[:,1:] - peak[1:-1][None,:] # compute Euclidean distance.
            peak_dist = np.sqrt(peak_dist[:,0]**2 + peak_dist[:,1]**2)

            min_peak_id = np.argmin(peak_dist) # which GT peak is closest to the predicted in same image/document.
            
            # has to be within the distance thresh
            if peak_dist[min_peak_id] <= dist_threshold:
                if relevant_match_count_GT[min_peak_id] == 0:
                    # true match (unique)
                    tp[ii] = 1
                    # update the GT match count
                    match_count_GT[select_relevant[min_peak_id]] = 1 # add one match. # already matched.   
                else:
                    # false match (non-unique)
                    fp[ii] = 1
            else:
                fp[ii] = 1
        
    # plot the curve:
    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val
        
    # form the cumulative recall
    rec = tp[:].copy()
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / float(len(all_gt_detections))
    
    # form the cumulative precision
    prec = tp[:].copy()
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    """
    append appropriate 0 and 1 to get the final curves for plotting.
    """
    ap, mrec, mprec = voc_ap(rec, prec)

    prec = np.insert(prec,0,1); prec = np.append(prec, 0)
    rec = np.insert(rec,0,0); rec = np.append(rec, 1)
    
    return ap, prec, rec
    

"""
TO DO: eval_MAD, eval_correlation -> given the image and annotations.
"""


if __name__=="__main__":
    
    from keras.models import load_model
    from skimage.exposure import rescale_intensity
    import numpy as np 
    import scipy.io as spio
    import pylab as plt 
    
    # input training files. 
#==============================================================================
#   Load the train-test_data
#==============================================================================
    # these contain the real distances too!. 
    in_files = ['Training_Testing_patches_sectioned-Early.mat',
                'Training_Testing_patches_sectioned-Mid.mat',
                'Training_Testing_patches_sectioned-Late.mat']

# =============================================================================
#   Load the CNN to test.
# =============================================================================
#    cnn_spot_model_early = load_model('Single-S1_32x32_relu_all_sigma2_mse')
#    cnn_spot_model_mid = load_model('Single-S1_32x32_relu_all_sigma2_mse')
#    cnn_spot_model_late = load_model('Single-S1_32x32_relu_all_sigma2_mse')
    
#    cnn_model = load_model('/media/felix/Elements/Raff Lab/Centriole Distancing/Scripts/sectioned_models/model_patch_32x32_aug_sigma2_raff_RB_mse_early')
#    cnn_spot_model_early = load_model('Multi-S1_32x32_relu_all_sigma2_mse')
#    cnn_spot_model_mid = load_model('Multi-S1_32x32_relu_all_sigma2_mse')
#    cnn_spot_model_late = load_model('Multi-S1_32x32_relu_all_sigma2_mse')
    
#    cnn_spot_model_early = load_model('Multi-S1_32x32_selu_all_sigma2_mse-2')
#    cnn_spot_model_mid = load_model('Multi-S1_32x32_selu_all_sigma2_mse-2')
#    cnn_spot_model_late = load_model('Multi-S1_32x32_selu_all_sigma2_mse-2')
    
#
#    cnn_spot_model_early = load_model('Single-S1_32x32_selu_all_sigma2_mse')
#    cnn_spot_model_mid = load_model('Single-S1_32x32_selu_all_sigma2_mse')s
#    cnn_spot_model_late = load_model('Single-S1_32x32_selu_all_sigma2_mse')
    
    
    """
    Sep Models.
    """
#    cnn_spot_model_early = load_model('Single-S1_32x32_relu_all_sigma2_mse-Early-notestaug')
#    cnn_spot_model_mid = load_model('Single-S1_32x32_relu_all_sigma2_mse-Mid-notestaug')
#    cnn_spot_model_late = load_model('Single-S1_32x32_relu_all_sigma2_mse-Late-notestaug')
    
#    cnn_spot_model_early = load_model('Single-S1_32x32_selu_all_sigma2_mse-Early-notestaug')
#    cnn_spot_model_mid = load_model('Single-S1_32x32_selu_all_sigma2_mse-Mid-notestaug')
#    cnn_spot_model_late = load_model('Single-S1_32x32_selu_all_sigma2_mse-Late-notestaug')
    
#    cnn_spot_model_early = load_model('Multi-S2_32x32_selu_all_sigma2_mse-Early-notestaug')
#    cnn_spot_model_mid = load_model('Multi-S2_32x32_selu_all_sigma2_mse-Mid-notestaug')
#    cnn_spot_model_late = load_model('Multi-S2_32x32_selu_all_sigma2_mse-Late-notestaug')
    
    cnn_spot_model_early = load_model('Multi-S1_32x32_selu_-attn-all_sigma2_mse-Early-notestaug')
    cnn_spot_model_mid = load_model('Multi-S1_32x32_selu_-attn-all_sigma2_mse-Mid-notestaug')
    cnn_spot_model_late = load_model('Multi-S1_32x32_selu_-attn-all_sigma2_mse-Late-notestaug')
    
# =============================================================================
#   Load all the data to test.
# =============================================================================
    setting = 'train'
    early_test = spio.loadmat(in_files[0])['X_test']
    early_test_GT = spio.loadmat(in_files[0])['Y_test']
    
    mid_test = spio.loadmat(in_files[1])['X_test']
    mid_test_GT = spio.loadmat(in_files[1])['Y_test']
    
    late_test = spio.loadmat(in_files[2])['X_test']
    late_test_GT = spio.loadmat(in_files[2])['Y_test']
    
    
    """
    Get GT measures including the positions of peaks. 
    """
    X_early, Y_early, dist_early, Peaks_early, Peaks_early_stack = annotations_to_dots_multi(early_test, early_test_GT)
    X_mid, Y_mid, dist_mid, Peaks_mid, Peaks_mid_stack = annotations_to_dots_multi(mid_test, mid_test_GT)
    X_late, Y_late, dist_late, Peaks_late, Peaks_late_stack = annotations_to_dots_multi(late_test, late_test_GT)
    
    
#    plt.figure()
#    plt.imshow(X_early[0])
#    plt.plot(Peaks_early[0][:,0], Peaks_early[0][:,1], '.')
#    plt.show()
    
# =============================================================================
#   Predict the CNN maps
# =============================================================================
    """
    Predict with CNN 
    """
    X_early = np.concatenate([rescale_intensity(x)[None,:] for x in X_early], axis=0)
    X_mid = np.concatenate([rescale_intensity(x)[None,:] for x in X_mid], axis=0)
    X_late = np.concatenate([rescale_intensity(x)[None,:] for x in X_late], axis=0)
    
    CNN_early_test = cnn_spot_model_early.predict(X_early[:,:,:,None]/255.)/ 1000.
    CNN_mid_test = cnn_spot_model_mid.predict(X_mid[:,:,:,None]/255.)/ 1000.
    CNN_late_test = cnn_spot_model_late.predict(X_late[:,:,:,None]/255.)/ 1000.

    
## =============================================================================
##   Test directly the quality of Peak (mAP) of the centrioles detected vs the manual positions.  
## =============================================================================
    
    # check the quality of the detected peaks. # bypass the other stages? 
    CNN_early_peaks = fetch_CNN_peaks(CNN_early_test[:,:,:,0], X_early/255., min_distance=1, filt=True)
    CNN_mid_peaks = fetch_CNN_peaks(CNN_mid_test[:,:,:,0], X_mid/255., min_distance=1, filt=True)
    CNN_late_peaks = fetch_CNN_peaks(CNN_late_test[:,:,:,0], X_late/255., min_distance=1, filt=True)
    
    all_CNN_peaks_early = np.vstack([p for p in CNN_early_peaks if len(p)> 0])
    all_CNN_peaks_mid = np.vstack([p for p in CNN_mid_peaks if len(p)> 0])
    all_CNN_peaks_late = np.vstack([p for p in CNN_late_peaks if len(p)> 0])
    
    
    mAP_early, _, _ = compute_AP(Peaks_early_stack, all_CNN_peaks_early, dist_threshold=3)
    mAP_mid, _, _ = compute_AP(Peaks_mid_stack, all_CNN_peaks_mid, dist_threshold=2)
    mAP_late, _, _ = compute_AP(Peaks_late_stack, all_CNN_peaks_late, dist_threshold=2)
    
    print(mAP_early, mAP_mid, mAP_late)
    
    all_CNN_peaks_mid[:,0] = all_CNN_peaks_mid[:,0] + all_CNN_peaks_early[-1,0] + 1
    all_CNN_peaks_late[:,0] = all_CNN_peaks_late[:,0] + all_CNN_peaks_mid[-1,0] + 1
    all_CNN_peaks = np.vstack([all_CNN_peaks_early, all_CNN_peaks_mid, all_CNN_peaks_late])
    

    Peaks_mid_stack[:,0] = Peaks_mid_stack[:,0] + Peaks_early_stack[-1][0] + 1 
    Peaks_late_stack[:,0] = Peaks_late_stack[:,0] + Peaks_mid_stack[-1,0] + 1
    all_GT_peaks = np.vstack([Peaks_early_stack, Peaks_mid_stack, Peaks_late_stack])
    
    mAP,  precision, recall= compute_AP(all_GT_peaks, all_CNN_peaks, dist_threshold=2)
    
    print(mAP)
    
    plt.figure()
    plt.plot(recall, precision, 'ko-')
    plt.show()
    
    # compute the mAP. 

#    plt.figure()
#    plt.imshow(CNN_early_test[0,:,:,0])
#    plt.plot(CNN_early_peaks[0][:,2], CNN_early_peaks[0][:,1], 'ro')
#    plt.plot(Peaks_early[0][:,1], Peaks_early[0][:,0], 'go')
#    plt.show()
#    
# =============================================================================
#   Eval the distance discrepancy and Correlation
# =============================================================================
    
    n_bootstraps = 10
    
#    CNN_early_test_dists = predict_centrioles_CNN_GMM(X_early/255., CNN_early_test, min_distance=1, filt=True, dist_thresh=15, ratio_thresh=4)
#    CNN_mid_test_dists = predict_centrioles_CNN_GMM(X_mid/255., CNN_mid_test, min_distance=1, filt=True, dist_thresh=15, ratio_thresh=4)
#    CNN_late_test_dists = predict_centrioles_CNN_GMM(X_late/255., CNN_late_test, min_distance=1, filt=True, dist_thresh=15, ratio_thresh=4)
#    
    
    
    print('evaluating MAD')
    from scipy.stats import pearsonr, spearmanr
    
    means_Early = []
    means_Mid = []
    means_Late = []
    means_all = []
    
    for iteration in range(n_bootstraps):
        """
        Get distancing information
        """
#        CNN_early_test_dists = predict_centrioles_CNN_GMM_single(X_early/255., CNN_early_test, min_distance=1, filt=True, dist_thresh=15, ratio_thresh=4)
#        CNN_mid_test_dists = predict_centrioles_CNN_GMM_single(X_mid/255., CNN_mid_test, min_distance=1, filt=True, dist_thresh=15, ratio_thresh=4)
#        CNN_late_test_dists = predict_centrioles_CNN_GMM_single(X_late/255., CNN_late_test, min_distance=1, filt=True, dist_thresh=15, ratio_thresh=4)
        
        CNN_early_test_dists = predict_centrioles_CNN_GMM(X_early/255., CNN_early_test, min_distance=1, filt=True, dist_thresh=15, ratio_thresh=4)
        CNN_mid_test_dists = predict_centrioles_CNN_GMM(X_mid/255., CNN_mid_test, min_distance=1, filt=True, dist_thresh=15, ratio_thresh=4)
        CNN_late_test_dists = predict_centrioles_CNN_GMM(X_late/255., CNN_late_test, min_distance=1, filt=True, dist_thresh=15, ratio_thresh=4)
        
        
        CNN_early_dists = np.hstack([p[1] for p in CNN_early_test_dists])
        CNN_mid_dists = np.hstack([p[1] for p in CNN_mid_test_dists])
        CNN_late_dists = np.hstack([p[1] for p in CNN_late_test_dists])
        
        all_CNN_dists = np.hstack([CNN_early_dists, CNN_mid_dists, CNN_late_dists])
        all_man_dists = np.hstack([dist_early, dist_mid, dist_late])
        
        means_Early.append([np.mean(np.abs(CNN_early_dists-dist_early)), np.std(np.abs(CNN_early_dists-dist_early))])
        means_Mid.append([np.mean(np.abs(CNN_mid_dists-dist_mid)), np.std(np.abs(CNN_mid_dists-dist_mid))])
        means_Late.append([np.mean(np.abs(CNN_late_dists-dist_late)), np.std(np.abs(CNN_late_dists-dist_late))])
        means_all.append([np.mean(np.abs(all_CNN_dists-all_man_dists)), np.std(np.abs(all_CNN_dists-all_man_dists)), pearsonr(all_man_dists, all_CNN_dists)[0]])
        
    print(means_Early)
    means_Early = np.mean(np.vstack(means_Early), axis=0)
    means_Mid = np.mean(np.vstack(means_Mid), axis=0)
    means_Late = np.mean(np.vstack(means_Late), axis=0)
    means_all = np.mean(np.vstack(means_all), axis=0)
    
    print('Early:', means_Early)
    print('Mid:', means_Mid)
    print('Late:', means_Late)
    print('Overall:', means_all)
    
#    print('Early:', np.mean(np.abs(CNN_early_dists-dist_early)), np.std(np.abs(CNN_early_dists-dist_early)))
#    print('Mid:', np.mean(np.abs(CNN_mid_dists-dist_mid)), np.std(np.abs(CNN_mid_dists-dist_mid)))
#    print('Late:', np.mean(np.abs(CNN_late_dists-dist_late)), np.std(np.abs(CNN_late_dists-dist_late)))
#    print('Overall:', np.mean(np.abs(all_CNN_dists-all_man_dists)), np.std(np.abs(all_CNN_dists-all_man_dists)))
    
    
##    print('Mean Early Man:', np.mean( dist_early))
##    print('Mean Mid Man:', np.mean(dist_mid))
##    print('Mean Late Man:', np.mean(dist_late))
#    
#    from scipy.stats import pearsonr, spearmanr
#    print('Pearson r:', pearsonr(all_man_dists, all_CNN_dists))
#    print('Spearman r:', spearmanr(all_man_dists, all_CNN_dists))
    
    
    
    
    
    
    
    
    
    