#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module contains functions that act on the output images of the trained CNN to carry out centriole distancing.

"""

import warnings
import numpy as np 
import image_fn

def associate_peaks2centre_single(cnn_peaks, cnn_centre, dist_thresh=10, ratio_thresh=3):
    """ Uses separation centre in order to return the 1 or 2 most likely centriole positions in the local patch.

    Parameters
    ----------
    cnn_peaks : numpy array or list 
        array of (y,x) coordinates of potential centriole centroids.
    cnn_centre : 2-tuple
        the coordinate of the separation centre of the centroid pair.
    dist_thresh : float
        two peaks separated by a distance > dist_thresh is unlikely to be mother-daughter centriole pairs. units are pixels
    ratio_thresh : float
        ratio of less intense over more intense centriole in a pair, < ratio_thresh, the pair is designated not similar in pixel intensity and only the brighter centriole is returned.
        
    Returns
    -------
    cnn_peaks_filt : numpy array
        (y,x) coordinates of the individual detected centriole centroids.

    """
    from sklearn.metrics.pairwise import pairwise_distances
    
    if len(cnn_peaks) > 2:
        # if more than 2 peaks predicted by CNN then find only the two peaks closest to the predicted centre.
        # can further improve this by insisting they lie on the same line. 
        dists = pairwise_distances(cnn_peaks[:,:2], cnn_centre)
        filt_peaks = cnn_peaks[np.argsort(np.squeeze(dists))[:2], :]
                               
        dist_pair = np.linalg.norm(filt_peaks[0,:2] - filt_peaks[1,:2])
        max_peak_pair = np.argsort(filt_peaks[:,2])
        peak_ratio = cnn_peaks[max_peak_pair[0],2] / cnn_peaks[max_peak_pair[1],2]
        
        if dist_pair >= dist_thresh or peak_ratio < 1./ratio_thresh: # this distance criteria is probably still best somewhat. 
            cnn_peaks_filt = filt_peaks[max_peak_pair[1],:2][None,:]
            return cnn_peaks_filt
        else:
            cnn_peaks_filt = filt_peaks[:,:2]
            return cnn_peaks_filt

    elif len(cnn_peaks) <= 2:
        if len(cnn_peaks) == 2: 
            # test the intensity of the cnn_peaks and the distance.
            dist_pair = np.linalg.norm(cnn_peaks[0,:2] - cnn_peaks[1,:2])
            max_peak_pair = np.argsort(cnn_peaks[:,2])
            peak_ratio = cnn_peaks[max_peak_pair[0],2] / cnn_peaks[max_peak_pair[1],2]
            
            if dist_pair >= dist_thresh  or peak_ratio < 1./ratio_thresh:
                cnn_peaks_filt = cnn_peaks[max_peak_pair[1],:2][None,:]
                return cnn_peaks_filt
            else:
                cnn_peaks_filt = cnn_peaks[:,:2]
                return cnn_peaks_filt
        elif len(cnn_peaks) == 1:
            cnn_peaks_filt = cnn_peaks[:,:2]
            return cnn_peaks_filt

        
def img2histogram_samples(img, thresh=0.1, samples=20):
    """ Efficient sampling of the (x,y) image coordinate pairs proportional to the corresponding pixel intensity.

    Parameters
    ----------
    img : numpy array 
        (n_rows x n_cols) gray-image
    thresh : float
        minimum threshold below which we set all intensity to 0 to avoid sampling very low intensities.
    samples : int
        the maximum number of times, a position (x,y) is sampled if I(x,y) = 1 that is maximum intensity

    Returns
    -------
    xy_samples : numpy array
        array of (x,y) coordinates whose distribution, :math:`p(x,y)\propto I(x,y)` where :math:`I(x,y)` is the corresponding (normalised) pixel intensity at (x,y).

    """
    from skimage.exposure import rescale_intensity

    nrows, ncols = img.shape
    # create coordinates.
    X, Y = np.meshgrid(range(ncols), range(nrows)); X = X.ravel(); Y = Y.ravel()

    # rescale image and set lower values to 0.
    im = rescale_intensity(img/255.)
    im[im<=thresh] = 0
    
    # sample the coordinates.
    im = (samples*im).astype(np.int)
    im = im.ravel()
    
    x_samples = np.hstack([im[i]*[X[i]] for i in range(len(im))])
    y_samples = np.hstack([im[i]*[Y[i]] for i in range(len(im))])
    xy_samples = np.vstack([x_samples, y_samples]).T 

    return xy_samples
        
def fit_2dGMM(samples_x, samples_y, n_components=2):
    """ Fit an n-component mixture to the 2d image to resolve closely overlapping centroids given (x,y) image coordinate samples

    samples_x and samples_y should be of same array length.

    Parameters
    ----------
    samples_x : numpy array 
        array of x-coordinates.
    samples_y : numpy array
        array of y-coordinates.
    n_components : int
        the number of Gaussian mixture components to fit, default is 2 to resolve the centriole pair.

    Returns
    -------
    gmm_params : 3-tuple
        Returns the fitted parameters in the mixture namely (weights, means, covariance matrices)
    prob_samples : numpy array 
        posterior probability of each sample belonging to each component. 
            input shape : (n_samples, 2) 
            output shape : (n_samples, n_components)
    gmm : Scikit-learn GMM model instance
        fitted Scikit-learn GMM model.

    """
    from sklearn.mixture import GaussianMixture
    
    samples_xy = np.vstack([samples_x, samples_y]).T
    
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(samples_xy)
    
    weights = gmm.weights_
    means_ = gmm.means_ 
    covars_ = gmm.covariances_
    gmm_params = (weights, means_, covars_)

    prob_samples = gmm.predict_proba(samples_xy)
    
    return gmm_params, prob_samples, gmm

def fitGMM_patch_post_process( centre_patch_intensity, n_samples=1000, max_dist_thresh=10, min_area_pair=0, max_area_pair=10000):
    """ Fits an n-component mixture to a 2d image to resolve closely overlapping centroids

    This function simplifies the calling and wraps `fit_2dGMM` so we directly give the input image. 

    Parameters
    ----------
    centre_patch_intensity : numpy array 
        input gray-image 
    n_samples : int
        the maximum number of samples to draw if the corresponding normalised image intensity was 1.
    max_dist_thresh : int
        the upper bound on the expected distance if it was a true pair.  

    Returns
    -------
    filt_peaks : 3-tuple
        returns the resolved centriole peak positions for distancing.

    """
    # thresholds mean + std. 
    thresh = np.mean(centre_patch_intensity) + np.std(centre_patch_intensity)

    # draw samples according to intensity 
    xy_samples = img2histogram_samples(centre_patch_intensity, thresh=thresh, samples=n_samples)
    
    if np.sum(xy_samples>=thresh) == 0:
        warnings.warn('not enough points with mean + std, trying mean threshold') 
        thresh = np.mean(centre_patch_intensity)
        xy_samples = img2histogram_samples(centre_patch_intensity, thresh=thresh, samples=n_samples)
    
    # fit the sample to GMM. 
    (weights, means_, covars_), fitted_y, gmm = fit_2dGMM(xy_samples[:,0], xy_samples[:,1], component=2)
    
    # get the centroids. 
    coords1 = means_[0]
    coords2 = means_[1]

    cand_pair_peaks = np.vstack([coords1, coords2]); cand_pair_peaks = cand_pair_peaks[:,[1,0]]
    cand_pair_dist = np.linalg.norm(cand_pair_peaks[0]-cand_pair_peaks[1])
    
    filt_peaks_backup = cand_pair_dist.copy()
    if cand_pair_dist <= max_dist_thresh:
        filt_peaks = cand_pair_peaks.copy()
    else:
        binary = centre_patch_intensity >= thresh        
        binary_filt = image_fn.filter_masks( binary, min_area=min_area_pair, max_area=max_area_pair, keep_centre=True, dist_thresh=1., min_max_area_cutoff=20)
        
        centre_patch_intensity_new = binary_filt*centre_patch_intensity
        
        # now do the detection 
        xy_samples = img2histogram_samples(centre_patch_intensity_new, thresh=thresh, samples=n_samples)
        (weights, means_, covars_), fitted_y, gmm = fit_2dGMM(xy_samples[:,0], xy_samples[:,1], component=2)
        
        coords1 = means_[0]
        coords2 = means_[1]
    
        cand_pair_peaks = np.vstack([coords1, coords2]); cand_pair_peaks = cand_pair_peaks[:,[1,0]]
        filt_peaks = cand_pair_peaks.copy()
        
        if len(filt_peaks) < 2:
            # return the original 2 peaks. 
            filt_peaks = filt_peaks_backup.copy()
    
    return filt_peaks


def detect_2d_peaks(img, I_img, min_distance=1, filt=True, thresh=None):
    """ Identify the centroids in the CNN predicted output images and return detected positions and intensity statistics. 

    Parameters
    ----------
    img : numpy array
        the CNN output probability image
    I_img : numpy array
        the raw gray-image 
    min_distance : float
        minimum separation distance between centroids in number of pixels.
    filt : bool
        if True, consider only peaks in the CNN output that have intensity >= thresh. If False, we use a small fixed threshold of 1e-6. 
    thresh : None or float
        if float, a constant threshold is used else Otsu thresholding is use to automatically set an intensity threshold.

    Returns
    -------
    n_peaks_raw : int
        Number of peaks detected without applying any intensity threshold.
    filt_peaks : array-like
        result array of (y,x,I) tuple of (y,x) centroid locations and associated raw pixel intensity at (y,x). If no peak is detected an empty list [] is returned. 

    """
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    
    peaks = peak_local_max(img, min_distance=min_distance)
    n_peaks_raw = len(peaks)
    
    if len(peaks) > 0: 
        
        I_peak = img[peaks[:,0], peaks[:,1]] # take the peak intensity from the image. 
        I_img_peak = I_img[peaks[:,0], peaks[:,1]]
        
        if filt == True:
            if thresh is None:
                thresh = threshold_otsu(img) # 1e-6 is also a good choice. 
            I_peak_out = I_img_peak[I_peak >= thresh]
            peaks = peaks[I_peak >= thresh]

            if len(peaks) > 0:
                filt_peaks = np.hstack([peaks, I_peak_out[:,None]])
                return n_peaks_raw, filt_peaks
            else:
                filt_peaks = []
                return n_peaks_raw, filt_peaks
        else:
            thresh = 1e-6
            I_peak_out = I_img_peak[I_peak >= thresh]
            peaks = peaks[I_peak >= thresh]

            if len(peaks) > 0:
                filt_peaks = np.hstack([peaks, I_peak_out[:,None]])
                return n_peaks_raw, filt_peaks
            else:
                return n_peaks_raw, []
    else:
        filt_peaks = []
        return n_peaks_raw, filt_peaks


def detect_2d_peaks_stack(img_stack, I_img_stack, min_distance=1, filt=True, thresh=None):
    """ Wrapper for :meth:`detect_2d_peaks` extending it to work on a stack of input images. 

    Parameters
    ----------
    img_stack : numpy array
        array of the CNN output probability images
    I_img_stack : numpy array
        array of raw gray-images 
    min_distance : float
        minimum separation distance between centroids in number of pixels.
    filt : bool
        if True, consider only peaks in the CNN output that have intensity >= thresh. If False, we use a small fixed threshold of 1e-6. 
    thresh : None or float
        if float, a constant threshold is used else Otsu thresholding is use to automatically set an intensity threshold.

    Returns
    -------
    all_n_peaks_raw : int
        Number of peaks detected without applying any intensity threshold.
    all_filt_peaks : array-like
        result array of (y,x,I) tuple of (y,x) centroid locations and associated raw pixel intensity at (y,x). If no peak is detected an empty list [] is returned. 

    """
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    
    all_filt_peaks = []
    all_n_peaks_raw = []

    # iterate through
    for ii, img in enumerate(img_stack):
        I_img = I_img_stack[ii]
        n_raw_peaks_img, peaks_img = detect_2d_peaks(img, I_img, min_distance=min_distance, filt=filt, thresh=thresh)
        all_n_peaks_raw.append(n_raw_peaks_img)
        all_filt_peaks.append(peaks_img)

    all_n_peaks_raw = np.hstack(all_n_peaks_raw)

    return all_n_peaks_raw, all_filt_peaks


def predict_centrioles_CNN_GMM(imstack, cnnstack, min_distance=1, filt=True, p_thresh=None, dist_thresh=15, ratio_thresh=4, nsamples_GMM=1000, max_dist_thresh_GMM=10):
    """ Wrapper function to take the input image stack and the predicted CNN output image stack and return distances.

    Parameters
    ----------
    imstack : numpy array
        array of input images for distancing.
    cnnstack : numpy array
        array of CNN output probability images.
    min_distance : float
        minimum separation distance between detected centroids in number of pixels.
    filt : bool
        if True, consider only peaks in the CNN output that have intensity >= p_thresh. If False, we use a small fixed threshold of 1e-6. 
    p_thresh : None or float
        minimum intensity threshold to filter detected peaks in CNN output. If None uses a threshold derived from Otsu's thresholding
    dist_thresh : None or float
        the upper bound on the expected distance if it was a true pair.
    ratio_thresh : float
        ratio of more intense over less intense centriole in a pair, > ratio_thresh, the pair is designated not similar in pixel intensity and only the brighter centriole is returned.
    nsamples_GMM : 
    max_dist_thresh_GMM :

    Returns
    -------
    distances : array
        list of [(y,x), d] list of detected (y,x) centroids and distance between the detected positions for each image. Returns empty array [] if no peaks are present.

    """
    from skimage.feature import peak_local_max

    distances = []
    
    for ii in range(len(imstack)):
        
        zstack = imstack[ii]
        out = cnnstack[ii]

        # 1. locate centriole peaks. 
        n_cnn_peaks_raw, cnn_peaks = detect_2d_peaks(out[:,:,0], zstack, min_distance=min_distance, filt=filt, thresh=p_thresh)
        
        # if find detection:
        if len(cnn_peaks) > 0:
            # 2. locate centres between the points. 
            if out.shape[-1] > 1:
                cnn_centre = peak_local_max(out[:,:,1], num_peaks=1) # only the highest peak. 
            else:
                # centre of the image. 
                cnn_centre = np.array([[test_zstack.shape[0]//2, test_zstack.shape[1]//2]]) 
            
            # default to mid-point also if no peak is detected using CNN images. 
            if len(cnn_centre) == 0:
                cnn_centre = np.array([[zstack.shape[0]//2, zstack.shape[1]//2]])  # defaults to mid point 
            
            # 3. associate centres to filter centriole locations. # how many to associate? Different strategy. 
            cnn_peaks_filt = associate_peaks2centre_single(cnn_peaks, cnn_centre, dist_thresh=dist_thresh, ratio_thresh=ratio_thresh)
            
            # two cases: (if only one then lets run the GMM on the CNN.)
            if len(cnn_peaks_filt) == 1: 
                centre_patch_intensity = out[:,:, 0].copy() # use the CNN map as the cleaned up ver. 
                cnn_peaks_filt = fitGMM_patch_post_process( centre_patch_intensity, n_samples=nsamples_GMM, max_dist_thresh=max_dist_thresh_GMM)
                
            peak_dist = np.linalg.norm(cnn_peaks_filt[0] - cnn_peaks_filt[1]) # find the distance between the points. 
            distances.append([cnn_peaks_filt, peak_dist])
        else: 
            distances.append([])

    return distances


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
    
    
    
    
    
    
    
    
    
    