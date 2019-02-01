#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module contains scripts for image manipulation including denoising, enhancement and cropping functions

"""

import numpy as np

def uint16_2_uint8(vidstack):
    """ Casts any input image to be of uint8 type. 

    Note: Though named uint16, converts any input to uint8. We are just implicitly assuming with biological imaging uint16 input.

    Parameters
    ----------
    vidstack : numpy array
        an input image (any size) as a numpy array.

    Returns
    -------
    uint8_img : numpy array
        a numpy array of same size as input rescaled to be of uint8 (range [0,255]).

    """
    uint8_img = np.uint8(255.*(vidstack/float(np.max(vidstack))))
    
    return uint8_img

def rescale_intensity_stack(img_stack):
    """ rescales the intensity of a series of images given as a (n_imgs x n_rows x n_cols x channels) tensor such that it is [0,255] for uint8 and [0,1] for floats.

    Parameters
    ----------
    img_stack : numpy array
        an input image of 3 or 4 dimensions:
            (n_imgs x n_rows x n_cols): gray-image stack
            (n_imgs x n_rows x n_cols x 3): rgb-image stack

    Returns
    -------
    img_stack_rescale : numpy array
        intensity rescaled images with range [0,255] for uint8 and [0,1] for floats

    """
    from skimage.exposure import rescale_intensity
    img_stack_rescale = np.concatenate([rescale_intensity(im)[None,:] for im in img_stack], axis=0)
    
    return img_stack_rescale

def resize_img_stack(img_stack, shape=(256,256)):
    """ Resizes a series of images given as a (n_imgs x n_rows x n_cols x channels) tensor.

    Parameters
    ----------
    img_stack : numpy array
        an input image of 3 or 4 dimensions:
            (n_imgs x n_rows x n_cols): gray-image stack
            (n_imgs x n_rows x n_cols x 3): rgb-image stack
    shape : 2-tuple
        (row_size, col_size) tuple giving the desired output image dimension 

    Returns
    -------
    img_stack_new : numpy array
        a numpy array of resized input:
            (n_imgs x shape[0] x shape[1]): gray-image stack
            (n_imgs x shape[0] x shape[1] x 3): rgb-image stack

    """
    from skimage.transform import resize
    img_stack_new = []

    for im in imgs:
        img_stack_new.append(resize(im, output_shape=shape)[None,:])
    
    img_stack_new = np.concatenate(imgs_, axis=0)
    
    return img_stack_new

def denoise_zstack(zstack):
    
#    from skimage.restoration import denoise_wavelet
    from skimage.filters import gaussian
    
    stacked = []
    
    for z in zstack:
#        stacked.append(denoise_wavelet(z)[None,:])
        stacked.append(gaussian(z, sigma=3)[None,:])
        
    return np.vstack(stacked)


def perona_malik(img, iterations=10, delta=0.14, kappa=15):
    
    """ Runs Perona-Malik anisotropic on a given grayscale image.

    Parameters
    ----------
    img : numpy array
        (n_rows x n_cols) grayscale image.
    iterations : int
        Number of iterations to run the diffusion process. Higher gives smoother output.
    delta : float
        This is the time step :math:`\Delta t` in the diffusion equation. 
    kappa : float
        This regulates the sensitivity to edges in the Perona-Malik formulation.

    Returns
    -------
    filtered_img : numpy array
        The filtered output image. Same size as input of type float.

    References
    ----------
    .. [1] Perona, P et. al, "Anisotropic diffusion." Geometry-driven diffusion in computer vision. Springer, Dordrecht, 1994. 73-92.

    """

    from scipy import misc, ndimage
    import numpy as np 
    # center pixel distances
    dx = 1
    dy = 1
    dd = np.sqrt(2)
    
    u = img.copy()
    
    # 2D finite difference windows
    windows = [
        np.array(
                [[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float64
        ),
        np.array(
                [[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float64
        ),
    ]
    
    for r in range(iterations):
        # approximate gradients
        nabla = [ ndimage.filters.convolve(u, w) for w in windows ]
    
        # approximate diffusion function
        diff = [ 1./(1 + (n/kappa)**2) for n in nabla]

        # update image
        terms = [diff[i]*nabla[i] for i in range(4)]
        terms += [(1/(dd**2))*diff[i]*nabla[i] for i in range(4, 8)]
        u = u + delta*(sum(terms))

    # Kernel for Gradient in x-direction
    Kx = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
    )
    # Kernel for Gradient in y-direction
    Ky = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
    )
    # Apply kernels to the image
    Ix = ndimage.filters.convolve(u, Kx)
    Iy = ndimage.filters.convolve(u, Ky)
    
    # return norm of (Ix, Iy)
    filtered_img = np.hypot(Ix, Iy)
    
    return filtered_img
    

def crop_patches_from_img(zstack, centroids, width=25):
    """ Crop image patches from a given input image of given width at given (x,y) centroid coordinates.

    Float centroids are first cast into ints.

    Parameters
    ----------
    zstack : numpy array
        input (n_rows x n_cols x n_channels) numpy array.
    centroids : numpy array or list
        array of (y,x) centroid coordinates
    width : int (odd)
        size of cropped image patch is (width x width x n_channels)
    
    Returns
    -------
    zs : numpy array
        an array of cropped patches with length equal to the number of centroids.

    """
    zs = []

    for cent in centroids:
        cent = cent.astype(np.int)
        patch = zstack[:,cent[0]-width//2:cent[0]-width//2+width, cent[1]-width//2:cent[1]-width//2+width][None,:]
        zs.append(patch)
        
    zs = np.concatenate(zs, axis=0)
    
    return zs


def filter_masks( mask, min_area=10, max_area=300):
    
    from skimage.measure import label, regionprops
    
    labelled = label(mask)
    uniq_reg = np.unique(labelled)[1:]

    if len(uniq_reg) == 1:
        area = np.sum(mask)
        
        if (area > min_area) and (area < max_area):
            return mask 
        else:
            return np.zeros_like(mask)
            
    else:
    
        reg = regionprops(labelled)
        uniq_reg = np.unique(labelled)[1:]
        
        areas = []
    
        for re in reg:
            areas.append(re.area)
            
        largest_reg = uniq_reg[np.argmax(areas)]
                               
        cand_mask = labelled == largest_reg
        
        if np.sum(cand_mask) > min_area and np.sum(cand_mask) < max_area:
            return cand_mask
        else:
            return np.zeros_like(cand_mask)
    

def filter_masks_centre( mask, min_area=10, max_area=300):
    
    from skimage.measure import label, regionprops
    from skimage.filters import gaussian
    

    nrows, ncols = mask.shape
    labelled = label(mask)
    uniq_reg = np.unique(labelled)[1:]
    mask_centre = np.array([nrows/2, ncols/2])

    if len(uniq_reg) == 1:
        area = np.sum(mask)
        
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

        largest_reg = uniq_reg[np.argmin(centre_dist)]
        min_dist = centre_dist[np.argmin(centre_dist)]
                               
        if min_dist >= 1/2.* nrows/2:
            return np.zeros_like(mask)
        else:              
            cand_mask = labelled == largest_reg
            
            if np.sum(cand_mask) > min_area and np.sum(cand_mask) < max_area:
                return cand_mask
            else:
                return np.zeros_like(cand_mask)


def find_best_focus(zstack):
    """ Finds the best focus slice by finding the z-slice that maximises the signal-to-noise ratio given by coefficient of variation (CV).
    
    .. math:: CV = \sigma/\mu

    where :math:`\sigma` and :math:`\mu` are the standard deviation and mean of the slice pixel intensities.

    Parameters
    ----------
    zstack : numpy array
        an input (n_z x n_rows x n_cols) image.

    Returns
    -------
    best_focus_slice : int
        index of the z-slice of best focus.

    """
    focus_vals = [np.var(z) / (np.mean(z)+1e-8) for z in zstack]
    best_focus_slice = np.argmax(focus_vals)
                  
    return best_focus_slice

    
def find_best_focus_stacks(zstacks):
    """ Finds the best focus slice of a series of z-slice stacks and constructs an array composed of the best-focus slices.

    Parameters
    ----------
    zstacks : numpy array
        an input (n_stacks x n_z x n_rows x n_cols) image.

    Returns
    -------
    best_focus_imgs : numpy array
        a new numpy array (n_stacks x n_rows x n_cols) composed of the best-focus slices only.
    best_focus_slices : numpy array
        list of the index of the z-slice of best focus for each z-slice stack.

    """
    best_focus_imgs = []
    best_focus_slices = []

    for zstack in zstacks:
        
        best_slice = find_best_focus(zstack)
        best_focus_img = zstack[best_slice]
        
        best_focus_slices.append(best_slice) # the best slice is needed to provide the slice to retrieve in the original video. 
        best_focus_imgs.append(best_focus_img[None,:])
        
    best_focus_imgs = np.concatenate(best_focus_imgs, axis=0)
    best_focus_slices = np.hstack(best_focus_slices)
    
    return best_focus_imgs, best_focus_slices


def locate_centroids_simple(mask):
    
    """ Given an image, locates all centroids of connected components.

    Note: This function inherently assumes a threshold of 0 and dilation with disk kernel of 3.

    Parameters
    ----------
    mask : numpy array
        an input grayscale image.

    Returns
    -------
    centroids : numpy array
        an array of (y,x) coordinate pairs giving the peaks in the input image.

    """
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_dilation, disk
    
    centroids = []
    mask_ = mask>0
    mask_ = binary_dilation(mask_, disk(3))
    labelled = label(mask_)
    regions = regionprops(labelled)
    
    for reg in regions:
        y,x = reg.centroid
        centroids.append([y,x])
    centroids = np.array(centroids)

    return centroids

def produce_valid_img_mask(img, min_I=0.1, max_area=1000, dilation=3):
    """ Example Centriole images may have a ring of high pixel intensity of a much larger structure. This function is designed to identify such large continuous areas in order to filter detections.
     
    Parameters
    ----------
    img : numpy array
        an input grayscale image.
    min_I : float
        the lower threshold for identifying the bright intensity regions. Assumes normalised intensities i.e. image intensities should be between [0,1] 
    max_area : integer
        threshold for identifying 'large' region based on counting the number of pixels within the area.
    dilation : int
        size of the disk kernel used to postprocess and smoothen resulting binary segmentation.

    Returns
    -------
    invalid_regions : numpy array
        a binary image of either 0, 1 pixel intensities indicating the large regions of high intensity i.e. invalid centriole zones.

    """

    from scipy.ndimage.morphology import binary_fill_holes
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_dilation, disk
    
    thresh = threshold_otsu(img) # determines an Ostu threshold.
    
    if np.mean(img[img>thresh]) > min_I: # is there signal in the image? which is the lower / better threshold to use.
        binary = img > thresh
    else:
        binary = img > min_I # resort to the manual guidance.
        
    # connected component analysis to identify large areas of high intensity.
    labelled = label(binary)
    regions = regionprops(labelled)
    
    # initialise the mask
    invalid_regions = np.zeros(labelled.shape)
    
    for i in range(len(regions)):
        area = regions[i].area
        # is it large?, if yes 
        if area > max_area:
            invalid_regions[labelled==i+1] = 1 # mark areas that satisfy the check to background
            
    invalid_regions = binary_dilation(binary_fill_holes(invalid_regions>0), disk(dilation)) # dilation is to smooth edges.

    return invalid_regions
    
def filter_noise_centroids_detection(centroids, mask):
    """ Given (y,x) coordinates and a binary mask of 0,1 of background regions, removes coordinates that lie in 1 areas (background).
     
    Parameters
    ----------
    centroids : numpy array
        array of (y,x) 2D coordinates.
    mask : numpy array
        boolean or integer mask with values 1 or 0 denoting invalid and valid spatial regions respectively.

    Returns
    -------
    filtered_centroids : numpy array
        array of only valid (y,x) 2D coordinates that lie in mask==0 regions.
    select : bool array
        a binary array either 0 or 1 indicating which centroids are valid.

    """
    valid_mask = mask[centroids[:,0].astype(np.int), centroids[:,1].astype(np.int)] #(y,x) format
    filtered_centroids = centroids[valid_mask==0]
    select = valid_mask == 0

    return filtered_centroids, select
    
def filter_border_centroids_detection(centroids, size, limits):
    """ Given (y,x) coordinates and the size of the border, removes all coordinates that lie within the defined border.
     
    Parameters
    ----------
    centroids : numpy array
        array of (y,x) 2D coordinates.
    size : int
        border size, how many pixels from the image edge do you consider the border. Isotropic border is assumed.
    limits : tuple-like
        (y_max, x_max) pair that define the maximum number of rows, columns respectively of the image.

    Returns
    -------
    filtered_centroids : numpy array
        array of only valid (y,x) 2D coordinates that do not lie in the border zone.
    select : bool array
        a binary array either 0 or 1 indicating which centroids lie within the border zone.
    
    """
    select_y = np.logical_and(centroids[:,0] > size, centroids[:,0] < limits[0]-size)
    select_x = np.logical_and(centroids[:,1] > size, centroids[:,1] < limits[1]-size)
    
    filtered_centroids = centroids[ np.logical_and(select_x, select_y)]
    select = np.logical_and(select_x, select_y)

    return filtered_centroids, select
 
def filter_centrioles_BCV(centroids, max_slice_im, patch_size, CV_thresh=0.3):
    """ Given (y,x) centroid coordinates, the maximum slice whole frame image filter detections based on signal-to-noise (SNR) ratio within local image crops. 

    The SNR measure used is the coefficient of variation, :math:`\sigma/\mu` where :math:`\sigma` and :math:`\mu` are the standard deviation and mean of the pixel intensities in the image patch.

    Parameters
    ----------
    centroids : numpy array
        array of (y,x) 2D coordinates.
    max_slice_im : numpy array
        a grayscale 2D image
    patch_size : int (odd)
        width of the local area to crop around the given (y,x) centroid
    CV_thresh : float
        Signal-to-noise ratio cut-off where SNR is measured by CV i.e. centroids are kept if :math:`CV>` CV_thresh

    Returns
    -------
    filtered_centroids : numpy array
        array of only valid (y,x) 2D coordinates that have :math:`CV>` CV_thresh.
    select : bool array
        a binary array either 0 or 1 indicating which centroids have :math:`CV>` CV_thresh.
    filtered_CV : array
        array with the corresponding CV of filtered_centroids.

    """
    # signal (biological coefficient of variation filter)
    patches = crop_patches_from_img(centrioles, max_slice_im, size=patch_size)
    snr_patches = np.hstack([np.std(p)/np.mean(p) for p in patches])
    
    # filter out the bogus detections? 
    select = snr_patches >= CV_thresh
    filtered_centroids = centrioles[select]
    filtered_CV = snr_patches[select]
    
    return filtered_centroids, select, filtered_CV


def remove_duplicate_centrioles(centroids, min_dist, lam=1000):
    """ Removes duplicate (y,x) returning only one (y,x) instance given array of (y,x) centroid coordinates and a minimum distance threshold below which we call two (y,x) duplicates, 

    Parameters
    ----------
    centroids : numpy array
        array of (y,x) 2D coordinates.
    min_dist : float
        two (y,x) coordinates are a duplicate if the distance between them is less than mid_dist.
    lam : float
        a very large float, typically just a number larger than the image diagonal to exclude oneself in the pairwise pairing process of (y,x) coordinates.
    
    Returns
    -------
    filtered_centroids : numpy array
        array of unique (y,x) 2D coordinates.
    select : bool array
        a binary array either 0 or 1 indicating which centroids are taken as unique (y,x) instances.
    
    """
    from sklearn.metrics.pairwise import pairwise_distances
    
    dist_matrix = pairwise_distances(centroids)
    dist_matrix += np.diag(lam*np.ones(len(centroids))) # prevent self interaction.
    # initialisation.
    select_filter = np.ones(len(centroids))
    
    for i in range(len(dist_matrix)):
        
        if select_filter[i] == 1:
            dist = dist_matrix[i]
            min_dist_arg = np.argmin(dist)

            if dist[min_dist_arg] < min_dist:
                select_filter[min_dist_arg] = 0 # set to false.

    select_filter = select_filter>0 # make binary
    filtered_centroids = centroids[select_filter>0]

    return filtered_centroids, select_filter

def detect_centrioles_in_img( zstack_img, size, aniso_params, patch_size, CV_thresh=0.3, tslice=0, is_img_slice=False, filter_border=True, filter_high_intensity_bg=True, remove_duplicates=True, filter_CV=True, separation=5, invert=False, minmass=10, minoverlap=10, bg_min_I=0.2, bg_max_area=1000, bg_dilation=3, bg_invalid_check=0.5, debug=False):
    """ Primary function that wraps various functions in this module into one API call to detect centrioles given an image or image stack.

    Parameters
    ----------
    zstack_img : numpy array
        either 
            i) a temporal z-stack (n_frames x n_z x n_rows x n_cols), 
            ii) a z-stack (n_z x n_rows x n_cols) or 
            iii) a grayscale image (n_rows x n_cols)
    size : float
        Approximate expected width of centriole to detect in image pixels.
    aniso_params : Python dict
        A Python dictionary giving the parameters for running the anisotropic filtering of Perona-Malik [1]_. This dictionary should contain the following keys: 'iterations', 'delta', kappa', see :meth:`image_fn.perona_malik`
    patch_size : int
        size of the local image patch to crop for filtering by CV if used, see :meth:`image_fn.filter_centrioles_BCV`
    CV_thresh : float
        coefficient of variation threshold for keeping high SNR detections as in :meth:`image_fn.filter_centrioles_BCV`
    tslice : int
        if tslice :math:`>=` 0, takes the corresponding time slice of the temporal z image and returns the max projection image over z. If zstack_img is just a zstack set tslice=-1.
    is_img_slice : bool
        Set True if input is a grayscale image.
    filter_border : bool
        If True, removes detections within a defined border zone
    filter_high_intensity_bg : bool
        If True, removes detections from high intensity background areas.
    remove_duplicates : bool
        If True, detects potential duplication of (y,x) locations that may by detecting the same centriole.
    filter_CV : bool 
        If True, keeps only (y,x) centriole detections whose CV evaluated over a local image crop is greater than a given threshold.
    separation : float
        minimum separation distance in pixels between blobs. 
    invert : bool
        if True, features of interest to detect are assumed darker than background, used in trackpy.locate, see [2]_
    minmass : float
        minimum integrated intensity values of detected blob used in trackpy.locate, see [2]_
    minoverlap : float
        distance threshold for calling duplicate (y,x) coordinates, see :meth:`image_fn.remove_duplicate_centrioles`
    bg_min_I : float 
        intensity cut-off for defining 'high' intensity image areas as in :meth:`image_fn.produce_valid_img_mask`
    bg_max_area : int
        area cut-off for defining 'large' background areas as in :meth:`image_fn.produce_valid_img_mask`
    bg_dilation : int 
        disk kernel size to dilate background noise mask as in :meth:`image_fn.produce_valid_img_mask`
    bg_invalid_check : float
        this is a check to prevent everything in the image being regarded as being invalid if one knows centrioles should be present. It is an upper bound on the total area of the invalid image area mask output of :meth:`image_fn.produce_valid_img_mask`.
    debug: bool
        if True, will produce all intermediate plotting graphics to help debugging.

    Returns
    -------
    out_dict : Python dict
        dictionary which collects the final output detections along with additional detection information.
        
        The dictionary has the following structure
            'centriole_centroids': 
                (y,x) coordinates of detected centrioles
            'centriole_pos': 
                table of all centriole detections with associated intensity statistics
            'max_proj_full_img': 
                maximum projection image
            'max_proj_full_img_denoise': 
                anisotropically filtered maximum projection image
            'background_mask': 
                background image area mask
            'valid_detection_mask': 
                non-background image areas where centrioles are being detected.
            'centriole_SNR': 
                associated :math:`CV` of detected centrioles

    References
    ----------
    .. [1] Perona, P et. al, "Anisotropic diffusion." Geometry-driven diffusion in computer vision. Springer, Dordrecht, 1994. 73-92.
    .. [2] TrackPy Gaussian blob detection, http://soft-matter.github.io/trackpy/dev/generated/trackpy.locate.html.
    
    """
    import trackpy as tp 
    from skimage.filters import threshold_otsu
    from skimage.exposure import rescale_intensity
    import visualization as viz
    import pylab as plt 

    ##########################################
    # 
    #   Handle different file inputs.
    #
    ##########################################
    if is_img_slice:
        if tslice >= 0:
            zstack_time_img = zstack_img[tslice].copy()
        else:
            zstack_time_img = zstack_img.copy()
        # max projection to detect positions. 
        slice_img = zstack_time_img.max(axis=0)
    else:
        slice_img = zstack_img.copy() # nothing to do.


    ##########################################
    #   Anisotropic filtering to enhance signal to background.
    ##########################################
    slice_img_denoise = rescale_intensity(perona_malik(rescale_intensity(slice_img/255.), iterations=aniso_params['iterations'], kappa=aniso_params['kappa'], delta=aniso_params['delta'])) # denoising, these parameters work well thus far for anisotropic diffusion. 
                
    ##########################################
    #   Gaussian blob detection (through TrackPy)
    ##########################################
    f = tp.locate(slice_img_denoise, size, separation=separation, invert=invert, minmass=minmass)

    if debug:
        """
        Viz 1 : initial detection
        """
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title('Initial Gaussian Blob Detection')
        plt.imshow(slice_img, cmap='gray')
        viz.draw_circles(np.vstack([f['y'], f['x']]).T, ax, radii=patch_size*1, col='r', lw=2)
        ax.grid('off')
        ax.axis('off')
        plt.show()
    
    ##########################################
    #   Precompute some binary masks for later (optional) use
    ##########################################
    valid_img_mask = produce_valid_img_mask(rescale_intensity(slice_img/255.), min_I=bg_min_I, max_area=bg_max_area, dilation=bg_dilation)
    background_img = slice_img < threshold_otsu(slice_img)

    """
    Optionally filter out border centriole detections
    """
    if filter_border:
        centriole_centroids = np.vstack([f['y'], f['x']]).T

        # filter the centroids ( don't care for those at the side. )
        centriole_centroids, centriole_centroids_filter = filter_border_centroids_detection(centriole_centroids, size=size, limits = slice_img.shape)
        
        f = f.iloc[centriole_centroids_filter]
        f.index = np.arange(len(centriole_centroids)) # re-index.

        if debug:
            """
            Viz 2 : Filter border detections. Border is highlighted with a yellow transparency mask.
            """
            border_mask = np.zeros((slice_img.shape[0], slice_img.shape[1], 3))
            border_mask[-size:,:, 0] = 1; border_mask[-size:,:, 1] = 1
            border_mask[:size, :, 0] = 1; border_mask[:size, :, 1] = 1
            border_mask[:,:size, 0] = 1; border_mask[:,:size, 1] = 1
            border_mask[:,-size:, 0] = 1; border_mask[:,-size:, 1] = 1
            
            fig, ax = plt.subplots(figsize=(10,10))
            plt.title('Filtering border detections')
            plt.imshow(slice_img, cmap='gray')
            plt.imshow(border_mask, alpha=0.6)
            viz.draw_circles(np.vstack([f['y'], f['x']]).T, ax, radii=patch_size*1, col='r', lw=2)
            ax.grid('off')
            ax.axis('off')
            plt.show()
            
    """
    Optionally filter out centriole detections in spurious large intensity band zones.
    """
    if filter_high_intensity_bg:
        if np.sum(valid_img_mask) / float(np.product(valid_img_mask.shape)) < bg_invalid_check: # check that not all the image is being highlighted as invalid.
            centriole_centroids, centriole_centroids_filter = filter_noise_centroids_detection(centriole_centroids, valid_img_mask)

            f = f.iloc[centriole_centroids_filter]
            f.index = np.arange(len(centriole_centroids)) # re-index.
            
            valid_img_mask = np.abs(1-valid_img_mask) >0 # invert since the valid_img_mask is really a background.
        else:
            valid_img_mask = np.ones_like(valid_img_mask)
            
        if debug:
            """
            Viz 3 : Filter background detections in spurious high intensity zones.
            """
            # compose a colour mask to highlight the invalid image regions
            color_slice_valid_mask = np.zeros([valid_img_mask.shape[0], valid_img_mask.shape[1], 3]); color_slice_valid_mask[:,:,0] = np.logical_not(valid_img_mask); color_slice_valid_mask[:,:,1] = np.logical_not(valid_img_mask)
    
            fig, ax = plt.subplots(figsize=(10,10))
            plt.title('Filtering high intensity regions')
            plt.imshow(slice_img, cmap='gray')
            plt.imshow(color_slice_valid_mask, alpha=0.6)
            viz.draw_circles(np.vstack([f['y'], f['x']]).T, ax, radii=patch_size*1, col='r', lw=2)
            ax.grid('off')
            ax.axis('off')
            plt.show()
    else:
        valid_img_mask = np.ones_like(valid_img_mask)

    """
    Remove duplicates.
    """
    if remove_duplicates:
        centriole_centroids, centriole_centroids_filter = remove_duplicate_centrioles(centriole_centroids, min_dist=minoverlap)
        f = f.iloc[centriole_centroids_filter]
        f.index = np.arange(len(centriole_centroids)) # re-index.

        if debug:
            """
            Viz 4 : Remove duplicate detections
            """
            fig, ax = plt.subplots(figsize=(10,10))
            plt.title('Removing duplicates by spatial proximity')
            plt.imshow(slice_img, cmap='gray')
            viz.draw_circles(np.vstack([f['y'], f['x']]).T, ax, radii=patch_size*1, col='r', lw=2)
            ax.grid('off')
            ax.axis('off')
            plt.show()
    
    """
    Remove low SNR.  
    """
    if filter_CV:
        # signal (biological coefficient of variation filter) [helps reduce false positives.]
        centriole_centroids, centriole_centroids_filter, centriole_SNR = filter_centrioles_BCV(centriole_centroids, slice_img, patch_size, CV_thresh=CV_thresh)
        f = f.iloc[centriole_centroids_filter]
        f.index = np.arange(len(centriole_centroids)) # re-index.

        if debug:
            """
            Viz 5 : Remove by CV
            """
            # final detection with white boxes 
            fig, ax = plt.subplots(figsize=(10,10))
            plt.title('Filtering by CV')
            plt.imshow(slice_img, cmap='gray')
            viz.draw_squares(np.vstack([f['y'], f['x']]).T, ax, width=patch_size, col='r', lw=2)
            ax.grid('off')
            ax.axis('off')
            plt.show()

    else:
        centriole_SNR = 0 # not computed.

    if debug:
            """
            Viz 6 : Final detections
            """
            # final detection with white boxes 
            fig, ax = plt.subplots(figsize=(10,10))
            plt.title('Final Detections')
            plt.imshow(slice_img, cmap='gray')
            viz.draw_squares(np.vstack([f['y'], f['x']]).T, ax, width=patch_size, col='r', lw=2)
            ax.grid('off')
            ax.axis('off')
            plt.show()
 
    out_dict = {'centriole_centroids':centriole_centroids, 
                'centriole_pos':f, 
                'max_proj_full_img':slice_img,
                'max_proj_full_img_denoise':slice_img_denoise,
                'background_mask':background_img,
                'valid_detection_mask':valid_img_mask,
                'centriole_SNR':centriole_SNR}

    return out_dict