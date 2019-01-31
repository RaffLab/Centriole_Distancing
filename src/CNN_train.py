#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This module contains functions to aid in parsing of manual dot annotations for centriole distancing and functions to prepare the data for CNN training.

"""

import numpy as np

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None, borderMode=cv2.BORDER_REFLECT_101):
    """Artificially augment the # of training images by elastic image transformations

    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5

    Parameters
    ----------
    image : numpy array
        input image, gray or RGB:
            (n_rows x n_cols): gray image
            (n_rows x n_cols x 3): RGB image
    alpha : float
        strength of deformation.
    sigma : float
        size of Gaussian filter for anti-aliasing. 
    sigma_affine : float
        size of the random local deformations 
    random_state : None or int
        integer seed for the random number generator, default: None
    borderMode :
        the border method used when extrapolating as given by cv2.warpAffine
        
    Returns
    -------
    out : numpy array
        the warped image of the same resolution.
            (n_rows x n_cols): a gray-image.
            (n_rows x n_cols x 3): an RGB-image. 
            (n_rows x n_cols x 4): an RGBA-image.
        
    References
    ----------
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    """
    import cv2
    from scipy.ndimage.interpolation import map_coordinates
    from scipy.ndimage.filters import gaussian_filter
    
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=borderMode)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    out = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    return out


def apply_elastic_transform(imgs, labels, strength=0.08, N=50):
    """This function wraps the elastic transform to apply it to a batch of images. 

    Parameters
    ----------
    imgs : numpy array
        array of input gray or RGB images:
            (n_imgs x n_rows x n_cols): gray image.
            (n_imgs x n_rows x n_cols x 3): RGB image.
    labels : numpy array
        array of corresponding annotation images for n different tasks, as represented by the number of image channels.
            (n_imgs x n_rows x n_cols x n_tasks): for n_tasks.
    strength : float
        the strength of the stretching in the elastic transform, see :meth:`elastic_transform` 
    N : int
        number of random deformations. 
    
    Returns
    -------
    aug_imgs : numpy array
        augmented image dataset, expanded N times. 
    aug_labels : numpy array
        corresponding annotation image dataset, expanded N times. 

    References
    ----------
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    """
    aug_imgs = []
    aug_labels = []
    n_imgs = len(imgs)    

    for i in range(n_imgs):
        im = imgs[i]
        lab = labels[i]

        im_ = np.dstack([im, lab])
        
        for j in range(N):
            im_out = elastic_transform(im_, im_.shape[1] * 2, im_.shape[1] * strength, im_.shape[1] * strength)
            aug_imgs.append(im_out[:,:,0][None,:])
            aug_labels.append(im_out[:,:,1][None,:])
            
    aug_imgs = np.concatenate(aug_imgs, axis=0)
    aug_labels = np.concatenate(aug_labels, axis=0)

    return aug_imgs, aug_labels
        
    
def random_intensity(img, shift=0.1):
    """Randomly adjust the image intensity using a gamma transform 

    Parameters
    ----------
    img : numpy array
        input gray or RGB images with intensity in [0,1]:
            (n_imgs x n_cols): gray image.
            (n_imgs x n_cols x 3): RGB image.
    shift : float
        the adjustment range of intensity. The intensity change is additively applied and is uniformly sampled from [-shift, shift]

    Returns
    -------
    im : numpy array
        output image of same size as input image with intensity in [0,1]

    """    
    # add random constant changes to brightness
    im = img + np.random.uniform(-shift, shift)
    im = np.clip(im, 0, 1)
    
    return im
    
    
def add_noise(img, shift, sigma):
    """Randomly add zero-mean Gaussian noise to the image + a random constant intensity change across the whole image. 

    Parameters
    ----------
    img : numpy array
        input gray or RGB images with intensity in [0,1]:
            (n_imgs x n_cols): gray image.
            (n_imgs x n_cols x 3): RGB image.
    shift : float
        the adjustment range of the constant intensity. The applied constant is uniformly sampled from [-shift, shift]
    sigma : float
        width of Gaussian, defines the noise level. 

    Returns
    -------
    im : numpy array
        output image of same size as input image with intensity in [0,1]

    """
    noise = np.random.normal(0, sigma, img.shape) # add zero mean Gaussian Noise. 
    im = random_intensity(img, shift=shift) + noise # also offset the intensity values a bit too. 
    im = np.clip(im, 0, 1) # check intensities are valid 
    
    return im 

def add_gamma(img, gamma=0.3):
    """Randomly adjust the image intensity using a gamma transform 

    Parameters
    ----------
    img : numpy array
        input gray or RGB images with intensity in [0,1]:
            (n_imgs x n_cols): gray image.
            (n_imgs x n_cols x 3): RGB image.
    gamma : numpy array
        the adjustment range of gamma intensity. The applied gamma is uniformly sampled from [1-gamma, 1+gamma]

    Returns
    -------
    im : numpy array
        output image of same size as input image with intensity in [0,1]

    """
    from skimage.exposure import adjust_gamma
    noise = np.random.uniform(1-gamma,1+gamma,1) # uniform sampling to get more diversity 
    im = adjust_gamma(img, gamma=noise, gain=1)
    im = np.clip(im, 0, 1) # clip to normalised intensities. 
    
    return im 
    
    
def apply_elastic_transform_intensity(imgs, labels, strength=0.08, shift=0.3, sigma_max=0.2, N=20, random_state=None):
    """This function wraps the elastic transform as well as adding random noise and gamma adjustment to augment a batch of images. 

    Parameters
    ----------
    imgs : numpy array
        array of input gray or RGB images:
            (n_imgs x n_cols): gray image.
            (n_imgs x n_cols x 3): RGB image.
    labels : numpy array
        array of corresponding annotation images for n different tasks, as represented by the number of image channels.
            (n_imgs x n_cols x n_tasks): for n_tasks.
    strength : float
        the strength of the stretching in the elastic transform, see :meth:`elastic_transform` 
    shift : float
        the maximum shift in pixel intensity in addition to addition of Gaussian noise. 
    sigma_max : float
        defines the maximum standard deviation of the Gaussian noise corruption. The noise level added is a uniform variable on the range [0, sigma_max]
    N : int
        number of random deformations. 
    random_state : int or None
        optionally set a random seed for the random generation.

    Returns
    -------
    aug_imgs : numpy array
        augmented image dataset, expanded N times. 
    aug_labels : numpy array
        corresponding annotation image dataset, expanded N times. 

    """
    from skimage.exposure import rescale_intensity
    aug_imgs = []
    aug_labels = []
    n_imgs = len(imgs)    

    for i in range(n_imgs):
        im = rescale_intensity(imgs[i])
        lab = labels[i]
        im_ = np.dstack([im, lab]) 
        if len(lab.shape) == 3:
            n_label_channels = lab.shape[-1]
        if len(lab.shape) == 2:
            n_label_channels = 1

        n_img_channels = im_.shape[-1] - n_label_channels
        
        for j in range(N):
            im_out = elastic_transform(im_, im_.shape[1] * strength, im_.shape[1] * strength, im_.shape[1] * strength, random_state=random_state)

            if multi==True:
            aug_imgs.append(im_out[:,:,:n_img_channels][None,:]) # no noise
            aug_labels.append(im_out[:,:,n_img_channels:n_img_channels+n_label_channels][None,:])  # with noise.             
            
            aug_imgs.append(add_noise(im_out[:,:,:n_img_channels], shift=shift, sigma=np.random.uniform(0,sigma_max,1))[None,:])
            aug_labels.append(im_out[:,:,n_img_channels:n_img_channels+n_label_channels][None,:]) 

            aug_imgs.append(add_gamma(im_out[:,:,:n_img_channels], gamma=0.3)[None,:]) # random gamma enhancement. 
            aug_labels.append(im_out[:,:,n_img_channels:n_img_channels+n_label_channels][None,:])

    aug_imgs = np.concatenate(aug_imgs, axis=0)
    aug_labels = np.concatenate(aug_labels, axis=0)

    return aug_imgs, aug_labels
    

def apply_gaussian_to_dots(img, sigma, min_I=0):
    """Given an image extract all unique annotation cases by matching the unique (R,G,B) colour used. 

    Parameters
    ----------
    img : numpy array
        (n_rows x n_cols) binary dot image.
    sigma : float
        width of Gaussian used to smooth the annotation. Should be roughly the size of the object being detected. 
    thresh : float
        intensity threshold to determine there is an annotation usually this is 0 for binary masks
    
    Returns
    -------
    im : numpy array
        Gaussian smoothed blob image. Sum of pixels should = number of objects.

    """
    from skimage.measure import label, regionprops
    from skimage.filters import gaussian
    
    binary = img > thresh
    labelled = label(binary)
    regions = regionprops(labelled)
    
    im = np.zeros_like(binary)
    for reg in regions:
        y,x = reg.centroid
        y = int(y)
        x = int(x)
        im[y,x] = 1
        
    im = gaussian(im, sigma=sigma, mode='reflect')
    
    return im 

def create_dot_annotations(xstack, ystack, sigma=5, min_I=0):
    """Given an image extract all unique annotation cases by matching the unique (R,G,B) colour used. 

    Parameters
    ----------
    xstack : numpy array
        array of input gray or RGB images:
            (n_imgs x n_rows x n_cols): gray image.
            (n_imgs x n_rows x n_cols x 3): RGB image.
    ystack : numpy array
        array of corresponding annotation images for n different tasks, as represented by the number of image channels.
            (n_imgs x n_cols x n_tasks): for n_tasks.
    sigma : float
        width of Gaussian used to smooth the annotation. Should be roughly the size of the object being detected. 
    min_I : int or float
        intensity threshold to determine there is an annotation usually this is 0 for binary masks
    
    Returns
    -------
    x_ : numpy array
        same array of input gray or RGB images matched to y_.
    y_ : numpy array 
        array of corresponding annotation images for n different tasks, as represented by the number of image channels with dot annotations now replaced with Gaussians.

    """
    from skimage.filters import threshold_otsu, gaussian
    from skimage.measure import label, regionprops

    y_ = []
    x_ = []
    
    for i in range(len(ystack)):
        im = ystack[i]
        dapi = xstack[i]

        if len(im.shape) == 2:
            if len(np.unique(im)) > 1: # its not blank
                out = apply_gaussian_to_dots(im, sigma, min_I=thresh)
                y_.append(im_out[None,:])
                x_.append(dapi[None,:])

        if len(im.shape) == 3:
            # multiple channels. 
            n_channels = im.shape[-1]
            im_out = []
            for j in range(n_channels):
                if len(np.unique(im[:,:,j])) > 1:
                    out = apply_gaussian_to_dots(im[:,:,j], sigma, min_I=thresh)
                    im_out.append(out)

            if len(im_out) == n_channels: # check all channels are represented. 
                im_out = np.dstack(im_out)
                y_.append(im_out[None,:])
                x_.append(dapi[None,:])
        
    y_ = np.concatenate(y_, axis=0)
    x_ = np.concatenate(x_, axis=0)
    
    return x_, y_
    
    
def extract_dots(img, color):
    """Given an image extract all unique annotation cases by matching the unique (R,G,B) colour used. 

    Parameters
    ----------
    img : numpy array
        an input RGB image or input RGB image stack
    color : tuple or list or numpy array
        (R,G,B) tuple to match
    
    Returns
    -------
    mask : bool numpy array
        binary image mask of matched colour:
            1. (n_rows x n_cols) for input RGB image
            2. (n_imgs x n_rows x n_cols) for input RGB image stack

    """
    if len(img.shape) == 3:
        mask_r = img[:,:,0] == color[0]
        mask_g = img[:,:,1] == color[1]
        mask_b = img[:,:,2] == color[2]

    if len(img.shape) == 4:
        mask_r = img[:,:,:,0] == color[0]
        mask_g = img[:,:,:,1] == color[1]
        mask_b = img[:,:,:,2] == color[2]
    
    mask = np.logical_and(mask_r, np.logical_and(mask_g, mask_b))
    
    return mask

def find_annot_centroids(labelled, method):
    """Given an integer method finds each individual object using different methods.

    Two methods are implemented:
        1) method = 'connected'
            uses connected component analysis to find unique objects
        2) method = 'local_peaks'
            uses idea of watershed to resolve objects when they overlap. 

    Parameters
    ----------
    labelled : numpy array
        an integer or binary thresholded image 
    method : string
        either 'connected' or 'local_peaks'
    
    Returns
    -------
    cents : numpy array
        array of (y,x) coordinates of found object centroids.

    """
    from skimage.measure import regionprops, label
    from skimage.filters import gaussian
    from skimage.feature import peak_local_max

    if method == 'connected':
        if np.max(labelled) == 1:
            labelled_ = label(labelled)
        else:
            labelled_ = labelled.copy()
        reg = regionprops(labelled_)
        
        cents = []
        for re in reg:
            y,x = re.centroid
            cents.append([y,x])
        cents = np.array(cents)

        return cents    
    
    if method == 'local_peak':
        im = gaussian(binary) # smooth first.
        cents = peak_local_max(im)
    
        return cents 
        
def annotations_to_dots(xstack, ystack, min_I=10):
    """Given image annotation, converts the annotation image to dot images where each dot is the centroid. 

    Parameters
    ----------
    xstack : numpy array
        array of input gray or RGB images:
            (n_imgs x n_rows x n_cols): gray image.
            (n_imgs x n_rows x n_cols x 3): RGB image.
    ystack : numpy array
        array of corresponding annotation images for n different tasks, as represented by the number of image channels.
            (n_imgs x n_cols x n_tasks): for n_tasks.
    min_I : int or float
        threshold cut-off for binarising annotation images. 
    
    Returns
    -------
    cells : numpy array
        matched input to `dots`.  
    dots :
        array same size as ystack with annotations converted to dots. 
    dists : 
        distance between manually marked centrioles 

    """
    from skimage.measure import label, regionprops
    from skimage.filters import gaussian
    
    cells = []
    dots = []
    dists = [] # should this be default?

    for i in range(len(ystack)):
        y = ystack[i]

        if len(y.shape) == 2:
            y = y[:,None] # pad extra channel

        n_rows, n_cols, n_channels = y.shape
        
        y_out = []
        dists_out = []

        for j in range(n_channels):
            labelled = label(y[:,:,j]>min_I) # threshold.
            n_regions = len(np.unique(labelled)) - 1
            
            if j == 0:
                if n_regions == 2:
                    # retrieve centroids of labelled regions. 
                    cents = find_annot_centroids(labelled, method='connected')
                else:
                    # it should be 1 and we use local peaks to retrieve.
                    cents = find_annot_centroids(y>min_I, method='local_peaks')
                    
                # check if centroids == 2. 
                if len(cents) == 2:
                    new_y = np.zeros((n_rows, n_cols), dtype=np.int)
                    cents = cents.astype(np.int)
                    for cent in cents:
                        new_y[cent[0], cent[1]] = 1
                    
                    y_out.append(new_y)
                    dists_out.append(np.linalg.norm(cents[0]-cents[1],2))

            elif j > 0: # for other annotation channels. assume only 1 dot within.
                cents = find_annot_centroids(labelled, method='connected')
                new_y = np.zeros((n_rows, n_cols), dtype=np.int)
                cents = cents.astype(np.int)
                if len(cents) == 1:
                    new_y[cents[:,0], cents[:,1]] = 1
                    y_out.append(new_y)

        if len(y_out) == n_channels:
            cells.append(xstack[i][None,:])
            dots.append(np.dstack(y_out)[None,:])
            dists.append(np.hstack(dists_out))

    cells = np.concatenate(cells, axis=0)
    dots = np.concatenate(dots, axis=0)
    dists = np.hstack(dists)

    return cells, dots, dists


def train_test_split(imgs, labels, split_ratio=0.8, seed=13337):
    """This function wraps the elastic transform to apply it to a batch of images. 

    Parameters
    ----------
    imgs : numpy array
        array of input gray or RGB images:
            (n_imgs x n_rows x n_cols): gray image.
            (n_imgs x n_rows x n_cols x 3): RGB image.
    labels : numpy array
        array of corresponding annotation images for n different tasks, as represented by the number of image channels.
            (n_imgs x n_rows x n_cols x n_tasks): for n_tasks.
    split_ratio : float
        the train-test split ratio. If total number of available images is N, a split_ratio of 0.8 results in 0.8*N:0.2*N distribution of train:test images.
    seed : int or None
        optional setting of the random number generator for reproducibility.
    
    Returns
    -------
    aug_imgs : numpy array
        augmented image dataset, expanded N times. 
    aug_labels : numpy array
        corresponding annotation image dataset, expanded N times. 
    """

    import numpy as np 
    if seed is not None:
        np.random.seed(seed)
    
    select = np.arange(imgs.shape[0])
    np.random.shuffle(select)
    
    n_train = int(split_ratio*len(select))
    
    train_x = imgs[select[:n_train]]
    train_y = labels[select[:n_train]]

    test_x = imgs[select[n_train:]]
    test_y = labels[select[n_train:]]

    return (train_x, train_y), (test_x, test_y)


if __name__=="__main__":
    
    import numpy as np 
    import pylab as plt 
    from tqdm import tqdm
    from keras.models import Model
    from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Concatenate, BatchNormalization, Activation
    from keras.optimizers import Adam
    from scipy.misc import imsave 
    import os 
    import cv2
    from skimage.exposure import rescale_intensity
    from skimage.filters import sobel
    import scipy.io as spio 
    
    np.random.seed(1674)
    
#==============================================================================
#   Load the files for training. 
#==============================================================================
    
    Stage = 'Late'
    
#    labelfolder = '../TrainingRB'
#    labelfolder = '../Sectioned Training RB/Early S-phase'
    labelfolder = '/media/felix/Elements/Raff Lab/Centriole Distancing/Sectioned Training RB/%s S-phase' %(Stage)
#    imgfolder = 'Patch_train_64x64'
    
    # first load in the image folder and then we can construct the specific names. 
    imgpaths, imgfiles = detect_img_files(labelfolder, key='.tif')
    
    n_imgs = len(imgfiles)
    
    X = []
    Y = []
    Z = []
    
    for i in tqdm(range(n_imgs)[:]):
        # read in the image. the 1st channel contains the raw..., the second contains the dot annotations. 
        im = read_multiimg_PIL(imgpaths[i])
        
        cells = im[:,:,:,0]
        dots = im[:,:,:,1]
        centres = im[:,:,:,2]

        for j in range(len(cells)):
            # iterate over the possible images. 
            if np.sum(dots[j])> 0:
                X.append(cells[j][None,:])
                Y.append(dots[j][None,:])
                Z.append(centres[j][None,:])
                
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    Z = np.concatenate(Z, axis=0)
    print '---\n'
    print 'number of total patches: %d' %(X.shape[0])
    
    Y_comb = np.concatenate([Y[:,:,:,None],Z[:,:,:,None]], axis=-1)
    
    X, Y_comb_dots, Dists = annotations_to_dots_multi(X, Y_comb)
    
    # create dots. 
    X, Y_comb_dots = create_dot_annotations_multi(X, Y_comb_dots, sigma=2, thresh=0)
    
    
    # sample out a fraction of the training data for testing purposes. (for self validation.)
    np.random.seed(13775)
    select = np.arange(len(X))
    np.random.shuffle(select)
    
    to_keep = int(.8*len(X))
    
    X_train = X[select[:to_keep]]
    Y_train = Y_comb[select[:to_keep]]
    D_train = Dists[select[:to_keep]]
    
    X_test = X[select[to_keep:]]
    Y_test = Y_comb[select[to_keep:]]
    D_test = Dists[select[to_keep:]]
    
    
    # save out (for safe keeping)
    spio.savemat('Training_Testing_patches_sectioned-%s.mat' %(Stage), {'X_train':X_train,
                                                                        'Y_train':Y_train,
                                                                        'D_train':D_train,
                                                                        'X_test':X_test,
                                                                        'Y_test':Y_test,
                                                                        'D_test':D_test})
    
    