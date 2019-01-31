#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 18:13:51 2019

@author: felix
"""

def associate_peaks2centre_single(cnn_peaks, cnn_centre, dist_thresh=10, ratio_thresh=3):
    
    """
    set additional filters:
    ------------------------
        1. intensity threshold ratio check between the two peaks detected. 
        2. distance threshold check between the two peaks detected. 
    """
    
    from sklearn.metrics.pairwise import pairwise_distances
    
    if len(cnn_peaks) > 2:
        dists = pairwise_distances(cnn_peaks[:,:2], cnn_centre)
        filt_peaks = cnn_peaks[np.argsort(np.squeeze(dists))[:2], :]
                               
        dist_pair = np.linalg.norm(filt_peaks[0,:2] - filt_peaks[1,:2])
        max_peak_pair = np.argsort(filt_peaks[:,2])
        peak_ratio = cnn_peaks[max_peak_pair[0],2] / cnn_peaks[max_peak_pair[1],2]
        
#        print dist_pair, peak_ratio
        if dist_pair >= dist_thresh or peak_ratio < 1./ratio_thresh: # this distance criteria is probably still best somewhat. 
            return filt_peaks[max_peak_pair[1],:2][None,:]
        else:
            return filt_peaks[:,:2]
        
#        return cnn_peaks[np.argsort(np.squeeze(dists))[:2], :2] # return the 2 closest peaks. 
        
    elif len(cnn_peaks) <= 2:
        
        if len(cnn_peaks) == 2: 
            # test the intensity of the cnn_peaks and the distance.
            dist_pair = np.linalg.norm(cnn_peaks[0,:2] - cnn_peaks[1,:2])
            max_peak_pair = np.argsort(cnn_peaks[:,2])
            peak_ratio = cnn_peaks[max_peak_pair[0],2] / cnn_peaks[max_peak_pair[1],2]
            
#            print dist_pair, peak_ratio
            if dist_pair >= dist_thresh  or peak_ratio < 1./ratio_thresh:
                return cnn_peaks[max_peak_pair[1],:2][None,:]
            else:
                return cnn_peaks[:,:2]
        elif len(cnn_peaks) == 1:
            return cnn_peaks[:,:2]
        
def img2histogram_samples(img, thresh=0.1, samples=20):

    from skimage.exposure import rescale_intensity
    # assuming 8 bit image.
    nrows, ncols = img.shape
    
    X, Y = np.meshgrid(range(ncols), range(nrows))
    X = X.ravel()
    Y = Y.ravel()

    im = rescale_intensity(img/255.)
    im[im<=thresh] = 0
    
#    plt.figure()
#    plt.imshow(im)
#    plt.show()
    
    im = (samples*im).astype(np.int)
    im = im.ravel()
    
    x_samples = np.hstack([im[i]*[X[i]] for i in range(len(im))])
    y_samples = np.hstack([im[i]*[Y[i]] for i in range(len(im))])
    
#    return xy_samples
    return np.vstack([x_samples, y_samples]).T  
      
def fitGMM_patch_post_process( centre_patch_intensity, n_samples=1000, max_dist_thresh=10):
    
    
    thresh = np.mean(centre_patch_intensity) + np.std(centre_patch_intensity)

    # attempt to do this on the full posterior graph. 
    xy_samples = img2histogram_samples(centre_patch_intensity, thresh=thresh, samples=n_samples)
    
    if np.sum(xy_samples>=thresh) == 0:
        print 'not enough points recalibrating thresholds' 
        thresh = np.mean(centre_patch_intensity)
        xy_samples = img2histogram_samples(centre_patch_intensity, thresh=thresh, samples=n_samples)
    
#    print np.sum(np.isnan(xy_samples))
    (weights, means_, covars_), fitted_y, gmm = fit_2dGMM(xy_samples[:,0], xy_samples[:,1], component=2)
    
    coords1 = means_[0]
    coords2 = means_[1]

    cand_pair_peaks = np.vstack([coords1, coords2]); cand_pair_peaks = cand_pair_peaks[:,[1,0]]
    cand_pair_dist = np.linalg.norm(cand_pair_peaks[0]-cand_pair_peaks[1])
    
    filt_peaks_backup = cand_pair_dist.copy()
    if cand_pair_dist <= max_dist_thresh:
        filt_peaks = cand_pair_peaks.copy()
    else:
        
#        print 'refining'
        # filter for the component closest to the centre of the image. 
        binary = centre_patch_intensity >= thresh
        
#        plt.figure()
#        plt.imshow(binary)
#        plt.show()
        
        binary_filt = filter_masks_centre( binary, min_area=0, max_area=10000)
        
##        plt.figure()
##        plt.imshow(binary)
#        plt.figure()
#        plt.imshow(binary_filt)
#        plt.show()
        
        centre_patch_intensity_new = binary_filt*centre_patch_intensity
        
#        plt.figure()
#        plt.imshow(centre_patch_intensity_new)
#        plt.show()
        
        # now do the detection 
        xy_samples = img2histogram_samples(centre_patch_intensity_new, thresh=thresh, samples=n_samples)
        (weights, means_, covars_), fitted_y, gmm = fit_2dGMM(xy_samples[:,0], xy_samples[:,1], component=2)
        
        coords1 = means_[0]
        coords2 = means_[1]
    
        cand_pair_peaks = np.vstack([coords1, coords2]); cand_pair_peaks = cand_pair_peaks[:,[1,0]]
        filt_peaks = cand_pair_peaks.copy()
        
        if len(filt_peaks) < 2:
            filt_peaks = filt_peaks_backup.copy()
    
    return filt_peaks
        
def fit_2dGMM(signature_x, signature_y, component=2):
    
    from sklearn.mixture import GaussianMixture
    
    signature = np.vstack([signature_x, signature_y]).T
    
    gmm = GaussianMixture(n_components=component)
    gmm.fit(signature)
    
    weights = gmm.weights_
    means_ = gmm.means_ 
    covars_ = gmm.covariances_

    fitted_y = gmm.predict_proba(signature)
    
    return (weights, means_, covars_), fitted_y, gmm


def filter_masks_centre( mask, min_area=10, max_area=300):
    
    from skimage.measure import label, regionprops
    from skimage.filters import gaussian
    

    nrows, ncols = mask.shape
    labelled = label(mask)

#    plt.figure()
#    plt.imshow(labelled)
#    plt.show()

    uniq_reg = np.unique(labelled)[1:]

    mask_centre = np.array([nrows/2, ncols/2])
#    print uniq_reg

    if len(uniq_reg) == 1:
        area = np.sum(mask)
        
#        print area
        if (area > min_area) and (area < max_area):
            return mask 
        else:
            return np.zeros_like(mask)
            
    else:
    
        reg = regionprops(labelled)
        uniq_reg = np.unique(labelled)[1:]
        
        areas = []
        centres = []
    
        for re in reg:
            y,x = re.centroid
            areas.append(re.area)
            centres.append([y,x])
            
        centres = np.array(centres)
        centre_dist = np.sqrt(np.sum((centres - mask_centre)**2, axis=1))
        
#        print centre_dist
#        print mask_centre
        # first take the nearest. 

        largest_reg = uniq_reg[np.argmin(centre_dist)]
        min_dist = centre_dist[np.argmin(centre_dist)]
                               
        if largest_reg <= 20:
            largest_reg = uniq_reg[np.argmax(areas)]
            min_dist = centre_dist[np.argmax(areas)]
                               
        if min_dist >= nrows:
            return np.zeros_like(mask)
#        largest_reg = uniq_reg[np.argmax(areas)]
        else:              
            cand_mask = labelled == largest_reg
            
            if np.sum(cand_mask) > min_area and np.sum(cand_mask) < max_area:
                return cand_mask
            else:
                return np.zeros_like(cand_mask)


def detect_2d_peaks(img, I_img, min_distance=1, filt=True):
    
    """
    If filt: then we apply threshold otsu on the CNN image.
    """
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    
    peaks = peak_local_max(img, min_distance=min_distance)
    n_peaks_raw = len(peaks)
    
    if len(peaks) > 0: 
        
        I_peak = img[peaks[:,0], peaks[:,1]] # take the peak intesnty from the image. 
        I_img_peak = I_img[peaks[:,0], peaks[:,1]]
        
        if filt == True:
            thresh = threshold_otsu(img)
#            thresh = 1e-6
            I_peak_out = I_img_peak[I_peak >= thresh]
            peaks = peaks[I_peak >= thresh]

            if len(peaks) > 0:
                return n_peaks_raw, np.hstack([peaks, I_peak_out[:,None]])
            else:
                return n_peaks_raw, []
        else:
            thresh = 1e-6
            I_peak_out = I_img_peak[I_peak >= thresh]
            peaks = peaks[I_peak >= thresh]

            if len(peaks) > 0:
                return n_peaks_raw, np.hstack([peaks, I_peak_out[:,None]])
            else:
                return n_peaks_raw, []
    else:
        return n_peaks_raw, []



def predict_centrioles_CNN_GMM(imstack, cnnstack, min_distance=1, filt=True, dist_thresh=15, ratio_thresh=4):
    
    from skimage.feature import peak_local_max
    """
    Post-filter results.
    """
    distances = []
    
    for ii in range(len(imstack)):
        
        zstack = imstack[ii]
        out = cnnstack[ii]
    
        # 1. locate centriole peaks. (there might be none? depending on the quality of initial detection?)
        n_cnn_peaks_raw, cnn_peaks = detect_2d_peaks(out[:,:,0], zstack, min_distance=min_distance, filt=filt)
        
        if len(cnn_peaks) > 0:
            # 2. locate centres between the points. 
            cnn_centre = peak_local_max(out[:,:,1], num_peaks=1) # only the highest peak. 
    #                cnn_centre = np.array([[test_zstack.shape[0]//2, test_zstack.shape[1]//2]]) 
            
            """
            can this be improved? 
            """
#            print(len(cnn_peaks), len(cnn_centre))
            
            if len(cnn_centre) == 0:
                cnn_centre = np.array([[zstack.shape[0]//2, zstack.shape[1]//2]])  # defaults to mid point 
            
            # 3. associate centres to filter centriole locations. # how many to associate? Different strategy. 
            cnn_peaks_filt = associate_peaks2centre_single(cnn_peaks, cnn_centre, dist_thresh=dist_thresh, ratio_thresh=ratio_thresh)
            
            # two cases: (if only one then lets run the GMM on the CNN.)
            if len(cnn_peaks_filt) == 1: 
                
#                print 'single'
                centre_patch_intensity = out[:,:, 0].copy() # use the CNN map as the cleaned up ver. 
    #                    centre_patch_intensity = test_zstack.copy()
                cnn_peaks_filt = fitGMM_patch_post_process( centre_patch_intensity, n_samples=1000, max_dist_thresh=10)
                
            peak_dist = np.linalg.norm(cnn_peaks_filt[0] - cnn_peaks_filt[1]) # find the distance between the points. 
            
            distances.append([cnn_peaks_filt, peak_dist])
            
        else: 
            distances.append([])
            
    return distances


def predict_centrioles_CNN_GMM_single(imstack, cnnstack, min_distance=1, filt=True, dist_thresh=15, ratio_thresh=4):
    
    from skimage.feature import peak_local_max
    """
    Post-filter results.
    """
    distances = []
    
    for ii in range(len(imstack)):
        
        zstack = imstack[ii]
        out = cnnstack[ii]
    
        # 1. locate centriole peaks. (there might be none? depending on the quality of initial detection?)
        n_cnn_peaks_raw, cnn_peaks = detect_2d_peaks(out[:,:,0], zstack, min_distance=min_distance, filt=filt)
        
        if len(cnn_peaks) > 0:
            # 2. locate centres between the points. 
#            cnn_centre = peak_local_max(out[:,:,1], num_peaks=1) # only the highest peak. 
            cnn_centre = np.array([[zstack.shape[0]//2, zstack.shape[1]//2]]) 
            
            """
            can this be improved? 
            """
#            print(len(cnn_peaks), len(cnn_centre))
            
            if len(cnn_centre) == 0:
                cnn_centre = np.array([[zstack.shape[0]//2, zstack.shape[1]//2]])  # defaults to mid point 
            
            # 3. associate centres to filter centriole locations. # how many to associate? Different strategy. 
            cnn_peaks_filt = associate_peaks2centre_single(cnn_peaks, cnn_centre, dist_thresh=dist_thresh, ratio_thresh=ratio_thresh)
            
            # two cases: (if only one then lets run the GMM on the CNN.)
            if len(cnn_peaks_filt) == 1: 
                
#                print 'single'
                centre_patch_intensity = out[:,:, 0].copy() # use the CNN map as the cleaned up ver. 
    #                    centre_patch_intensity = test_zstack.copy()
                cnn_peaks_filt = fitGMM_patch_post_process( centre_patch_intensity, n_samples=1000, max_dist_thresh=10)
                
            peak_dist = np.linalg.norm(cnn_peaks_filt[0] - cnn_peaks_filt[1]) # find the distance between the points. 
            
            distances.append([cnn_peaks_filt, peak_dist])
            
        else: 
            distances.append([])
            
    return distances


def annotations_to_dots_multi(xstack, ystack):
    
    from skimage.measure import label, regionprops
    from skimage.filters import gaussian
    
    cells = []
    dots = []
    dists = []
    peaks = []

    for i in range(len(ystack)):
        y = ystack[i]
        n_rows, n_cols, n_channels = y.shape
        y_out = []
        dists_out = []
        peaks_out = []

        for j in range(n_channels):
            labelled = label(y[:,:,j]>10) # threshold.
            n_regions = len(np.unique(labelled)) - 1
            
            if j == 0:
                if n_regions == 2:
                    # retrieve centroids of labelled regions. 
                    cents = ret_centroids_regionprops(labelled)
                else:
                    # it should be 1 and we use local peaks to retrieve.
                    cents = ret_centroids_localpeaks(y>10)
                    
                # check if centroids == 2. 
                if len(cents) == 2:
                    new_y = np.zeros((n_rows, n_cols), dtype=np.int)
                    cents = cents.astype(np.int)
                    for cent in cents:
                        new_y[cent[0], cent[1]] = 1
                    
                    y_out.append(new_y)
                    dists_out.append(np.linalg.norm(cents[0]-cents[1],2))
                    peaks_out.append(cents[None,:])
            elif j > 0: # for other annotation channels. 
                cents = ret_centroids_regionprops(labelled)
                new_y = np.zeros((n_rows, n_cols), dtype=np.int)
                cents = cents.astype(np.int)
                if len(cents) == 1:
                    new_y[cents[:,0], cents[:,1]] = 1
                    y_out.append(new_y)

        if len(y_out) == n_channels:
            cells.append(xstack[i][None,:])
            dots.append(np.dstack(y_out)[None,:])
            dists.append(np.hstack(dists_out))
            peaks.append(np.concatenate(peaks_out, axis=0))
            
    peak_stack = np.vstack([p[0] for p in peaks])
    peak_ids = np.hstack([[i]*len(peaks[i][0]) for i in range(len(peaks))])
    
#    print(peak_stack.shape, len(peak_ids))
    peak_stack = np.hstack([peak_ids[:,None], peak_stack])
           
    return np.concatenate(cells, axis=0), np.concatenate(dots, axis=0), np.hstack(dists), np.concatenate(peaks, axis=0), peak_stack


def ret_centroids_regionprops(labelled):
    
    from skimage.measure import regionprops
    reg = regionprops(labelled)
    
    cents = []
    
    for re in reg:
        y,x = re.centroid
        cents.append([y,x])
        
    return np.array(cents)
    
def ret_centroids_localpeaks(binary):
    
    from skimage.filters import gaussian
    from skimage.feature import peak_local_max
    
    im = gaussian(binary) # smooth them.
    peaks = peak_local_max(im)
    
    return peaks 


def fetch_CNN_peaks(img_stack, I_img_stack, min_distance=1, filt=True):
    
    """
    If filt: then we apply threshold otsu on the CNN image.
    """
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    
    all_peaks = []
    
    # iterate through
    for ii, _ in enumerate(img_stack):
    
        img = np.squeeze(img_stack[ii])
        peaks = peak_local_max(img, min_distance=min_distance)
        
        if len(peaks) > 0: 
            
            I_peak = img[peaks[:,0], peaks[:,1]] # take the peak intesnty from the image. 
    #        p_peak = I_img[peaks[:,0], peaks[:,1]]
            
            if filt == True:
#                thresh = threshold_otsu(img)
                thresh = 1e-6
                I_peak_out = I_peak[I_peak >= thresh]
                peaks = peaks[I_peak >= thresh]
    
                if len(peaks) > 0:
                    all_peaks.append( np.hstack([ ii*np.ones(len(peaks))[:,None], peaks, I_peak_out[:,None]]) )
                else:
                    all_peaks.append([])
        else:
            all_peaks.append([[]])
            
    return all_peaks


def voc_ap(rec, prec):
  """
  --- Official matlab code VOC2012---
  mrec=[0 ; rec ; 1];
  mpre=[0 ; prec ; 0];
  for i=numel(mpre)-1:-1:1
      mpre(i)=max(mpre(i),mpre(i+1));
  end
  i=find(mrec(2:end)~=mrec(1:end-1))+1;
  ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
  """
  rec.insert(0, 0.0) # insert 0.0 at begining of list
  rec.append(1.0) # insert 1.0 at end of list
  mrec = rec[:]
  prec.insert(0, 0.0) # insert 0.0 at begining of list
  prec.append(0.0) # insert 0.0 at end of list
  mpre = prec[:]
  """
   This part makes the precision monotonically decreasing
    (goes from the end to the beginning)
    matlab:  for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
  """
  # matlab indexes start in 1 but python in 0, so I have to do:
  #   range(start=(len(mpre) - 2), end=0, step=-1)
  # also the python function range excludes the end, resulting in:
  #   range(start=(len(mpre) - 2), end=-1, step=-1)
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])
  """
   This part creates a list of indexes where the recall changes
    matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
  """
  i_list = []
  for i in range(1, len(mrec)):
    if mrec[i] != mrec[i-1]:
      i_list.append(i) # if it was matlab would be i + 1
  """
   The Average Precision (AP) is the area under the curve
    (numerical integration)
    matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
  """
  ap = 0.0
  for i in i_list:
    ap += ((mrec[i]-mrec[i-1])*mpre[i])
  return ap, mrec, mpre

        
def compute_AP(all_gt_peaks, all_pred_peaks, dist_threshold=3):
    
    import pylab as plt 
    """
    solve information retrieval. 
    """
    dist_threshold = dist_threshold # 10 frames or 5 mins. 
    
    fp = np.zeros(len(all_pred_peaks)) # how many to get wrong, 
    tp = np.zeros(len(all_pred_peaks)) # how many we get right 
    n_gt = np.zeros(len(all_pred_peaks)) # how many total ground-truth there are.
    match_count_GT = np.zeros(len(all_gt_peaks))
    
    # rank the all_pred_peaks. 
    all_pred_peaks_sort = all_pred_peaks[all_pred_peaks[:,-1].argsort()[::-1]] # sort by confidence
    
#    plt.figure()
#    plt.plot(all_pred_peaks_sort[:,-1])
#    plt.show()
    
    for ii in range(len(all_pred_peaks))[:]:
        
        peak = all_pred_peaks_sort[ii]
        select_relevant = np.arange(len(all_gt_peaks))[all_gt_peaks[:,0] == peak[0]]
        relevant_gt_peaks = all_gt_peaks[select_relevant] # which ones we consider. 
        relevant_match_count_GT = match_count_GT[select_relevant]
        
#        n_gt[ii] = len(relevant_gt_peaks) # number of possible ground truths. 
        # attempt to match. 
        if len(relevant_gt_peaks) == 0:
            fp[ii] = 1 # false peak. 
        else:
#            peak_dist = np.abs(relevant_gt_peaks[:,0] - peak[0])
            peak_dist = relevant_gt_peaks[:,1:] - peak[1:-1][None,:]
            peak_dist = np.sqrt(peak_dist[:,0]**2 + peak_dist[:,1]**2)
            min_peak_id = np.argmin(peak_dist)
            
            # has to be within the distance thresh
            if peak_dist[min_peak_id] <= dist_threshold:
                if relevant_match_count_GT[min_peak_id] == 0:
                    # true match (unique)
                    tp[ii] = 1
                    # update 
#                    match_count_GT[select_relevant[min_peak_id]] = 1 # add one match. # already matched.   
                else:
                    # false match (non-unique)
                    fp[ii] = 1
            else:
                fp[ii] = 1
        
#    print(np.sum(tp))
    # plot the curve:
    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val
        
    #print(tp)
    rec = tp[:].copy()
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / float(len(all_gt_peaks))
    #print(rec)
    prec = tp[:].copy()
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    
    
    """
    append appropriate 0 and 1. 
    """
    prec = np.insert(prec,0,1); prec = np.append(prec, 0)
    rec = np.insert(rec,0,0); rec = np.append(rec, 1)
    
    ap, mrec, mprec = voc_ap(list(rec), list(prec))
    
    
    return ap, prec, rec
    

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
    
    
    
    
    
    
    
    
    
    