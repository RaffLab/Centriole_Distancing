#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:19:11 2017

@author: felix
"""

import numpy as np

def resize_img(imgs, shape=(256,256)):
    
    from skimage.transform import resize
    imgs_ = []

    for im in imgs:
        imgs_.append(resize(im, output_shape=shape)[None,:])
    
    imgs_ = np.concatenate(imgs_, axis=0)
    
    return imgs_
  
def apply_augmentations( img, annot, contrast=0, scale=[1./2, 2], rotate=np.pi/2., translate=0, shear=0.2, N=20):
    
    from skimage.transform import AffineTransform, warp, SimilarityTransform

    n_imgs = len(img)
    
    out_img = []
    out_annot = []
    
    for i in range(n_imgs):
        
        rand_scale = np.random.uniform(scale[0], scale[1], N)
        rand_rotate = np.random.uniform(-rotate, rotate, N)
        rand_translate = np.random.uniform(-translate, translate, N)
        rand_shear = np.random.uniform(-shear, shear, N)
        
        for j in range(N):

            tf = AffineTransform(scale=(rand_scale[j], rand_scale[j]), rotation=rand_rotate[j], shear=rand_shear[j], translation=rand_translate[j])

            shift_y, shift_x = np.array(img[i].shape[:2]) / 2.  
            tf_shift = SimilarityTransform(translation=[-shift_x, -shift_y])  
            tf_shift_inv = SimilarityTransform(translation=[shift_x, shift_y])
#            outimg_ = warpRGB(img[], tf)
#            outannot_ = warpRGB(annot, tf)
#            outimg_ = warp(img[i], tf)
#            outannot_ = warp(annot[i], tf)
            outimg_ = warp(img[i], (tf_shift + (tf + tf_shift_inv)).inverse)
            outannot_ = warp(annot[i], (tf_shift + (tf + tf_shift_inv)).inverse)
            
            out_img.append(outimg_)
            out_annot.append(outannot_)
            
    return out_img, out_annot
    
    
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
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
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def apply_elastic_transform(imgs, labels, strength=0.08, N=50):
    
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
            
    return np.concatenate(aug_imgs, axis=0), np.concatenate(aug_labels, axis=0)
        
    
def train_test_split(imgs, labels, split_ratio=0.8, seed=13337):
    
    import numpy as np 
    np.random.seed(seed)
    
    select = np.arange(imgs.shape[0])
    np.random.shuffle(select)
    
    n_train = int(split_ratio*len(select))
    
    train_x = imgs[select[:n_train]]
    train_y = labels[select[:n_train]]

    test_x = imgs[select[n_train:]]
    test_y = labels[select[n_train:]]

    return (train_x, train_y), (test_x, test_y)
    
def sample_patches(img, binary, patch_size=(32,32), sample_N=20):
    
    from skimage.morphology import binary_erosion, disk
    padsize = patch_size[0]/2
    padded = np.pad(img, 2*padsize, mode='constant')
    binary_ = np.pad(binary, 2*padsize, mode='constant')
    binary__ = np.pad(binary, 2*padsize, mode='constant')
#    binary__ = binary_erosion(binary__, disk(1))
    
    m, n = binary_.shape
    X, Y = np.meshgrid(range(n), range(m))
    
    valid_X = X[binary__]
    valid_Y = Y[binary__]
    np.random.shuffle(valid_X)
    np.random.shuffle(valid_Y)
    sample_x = valid_X[:sample_N]
    sample_y = valid_Y[:sample_N]

#    range_X = [np.min(valid_X), np.max(valid_X)]
#    range_Y = [np.min(valid_Y), np.max(valid_Y)]   
#    sample_x = np.random.uniform(range_X[0], range_X[1], sample_N)
#    sample_y = np.random.uniform(range_Y[0], range_Y[1], sample_N)
#    # adopt a gaussian sampling scheme.
#    sample_x = np.random.normal(int(.5*(np.min(valid_X)+np.max(valid_X))), (np.max(valid_X)-np.min(valid_X)), sample_N)
#    sample_y = np.random.normal(int(.5*(np.min(valid_Y)+np.max(valid_Y))), (np.max(valid_Y)-np.min(valid_Y)), sample_N)
#    sample_x = sample_x.astype(np.int)
#    sample_y = sample_y.astype(np.int)

#    plt.figure()
#    plt.subplot(121)
#    plt.imshow(padded)
#    plt.plot(sample_x, sample_y, 'r.')
#    plt.subplot(122)
#    plt.imshow(binary_, cmap='gray')
#    plt.show()
        
#    bboxes = []
    imgpatches = []
    annotpatches = []

    for i in range(len(sample_x)):
        xmin = sample_x[i] - padsize
        xmax = xmin + patch_size[0]
        ymin = sample_y[i] - padsize
        ymax = ymin + patch_size[1]

#        bboxes.append([xmin, ymin, xmax, ymax])
#    bboxes = np.vstack(bboxes)
        imgpatches.append(padded[ymin:ymax, xmin:xmax][None,:])
        annotpatches.append(binary_[ymin:ymax, xmin:xmax][None,:])
        
#    bboxes = np.vstack(bboxes)
    imgpatches = np.concatenate(imgpatches, axis=0)
    annotpatches = np.concatenate(annotpatches, axis=0)
    
    return imgpatches, annotpatches, np.vstack([sample_x, sample_y]).T


def sample_patches_aug(imglist, binarylist, patch_size=(32,32), sample_N=20):
    
    imgs = []
    annots = []

    for i in range(len(imglist)):
        im = imglist[i]
        bi = binarylist[i]

        s_patch_0, s_annot_0, _ = sample_patches(im, bi>10, patch_size=patch_size, sample_N=sample_N)
        imgs.append(s_patch_0)
        annots.append(s_annot_0)
        
    return imgs, annots
    
    
def draw_boxes(boxes, ax):
    
    for box in boxes:
        xmin, ymin, xmax, ymax = box

        ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], lw=2)
    
    return []

def mkdir(directory):
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return []


def apply_nn_patches(single_img, model, patch_size=32):

    from skimage.transform import resize
    from skimage.exposure import rescale_intensity
    from skimage.util import img_as_ubyte
    # cut up the shape into different patches...
    
    m, n = single_img.shape
    
    if m !=256 and n !=256:
#        im = resize(single_img, (256,256))
        im = np.zeros((512,512))
        im[:m, :n] = single_img.copy()
    else:
        im = single_img.copy()
        
#    plt.figure()
#    plt.imshow(im)
#    plt.show()
    
    m, n = im.shape
    
#    im = img_as_ubyte(im)
    canvas_out = np.zeros(im.shape, dtype=np.float)
        
    n_x = np.linspace(0, n - patch_size , 64).astype(np.int)
    n_y = np.linspace(0, m - patch_size , 64).astype(np.int)
    
    for i in n_y:
        for j in n_x:
#            print (i,j)
            patch = im[i:i+patch_size, j:j+patch_size] # cut out the patch 
            patch = rescale_intensity(patch / 255.)
#            patch = img_as_ubyte(patch)
            pred = model.predict(patch[None,:,:,None])
            pred = np.squeeze(pred)
#            print pred.shape
            # now cut out the border. 
#            canvas_out[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = pred[padding:-padding, padding:-padding]
            canvas_out[i:i+patch_size, j:j+patch_size] += pred 
                       
    return canvas_out
    
    
def random_intensity(img, shift=0.1):
    
    hsv = img.copy()

    # do the scalings & shiftings
    hsv[..., 0] += np.random.uniform(-shift, shift)

    # cut off invalid values
    hsv = np.clip(hsv, 0, 1)

    # round to full numbers
#    hsv = np.uint8(np.round(hsv * 255.))
    
    return hsv
    
def contrast_adjust(img, contrast_factor):
    
    im = (img - np.mean(img)) * contrast_factor + np.mean(img)
#    im = np.clip(im , 0, 1)
    
    return im
    
def add_noise(img, shift, sigma):
    
    noise = np.random.normal(0, sigma, img.shape)
#    noise = np.random.uniform(-sigma, sigma, img.shape)
    im = random_intensity(img, shift=shift) + noise

    im = np.clip(im, 0, 1)
    
    return im 

def add_gamma(img, gamma=0.3):
    
    from skimage.exposure import adjust_gamma
#    noise = np.random.normal(0, gamma, 1)
    noise = np.random.uniform(1-0.3,1+0.3,1)
    im = adjust_gamma(img, gamma=noise, gain=1)

    im = np.clip(im, 0, 1)
    
    return im 
    
    
    
def apply_elastic_transform_intensity(imgs, labels, strength=0.08, shift=0.3, N=20):
    
    from skimage.exposure import rescale_intensity
    aug_imgs = []
    aug_labels = []
    n_imgs = len(imgs)    

    for i in range(n_imgs):
        im = rescale_intensity(imgs[i])
        lab = labels[i]

        im_ = np.dstack([im, lab])
        
        for j in range(50):
            im_out = elastic_transform(im_, im_.shape[1] * strength, im_.shape[1] * strength, im_.shape[1] * strength, random_state=None)

            aug_imgs.append(im_out[:,:,0][None,:]) # no noise
            aug_labels.append(im_out[:,:,1][None,:])  # with noise.             
            aug_imgs.append(add_noise(im_out[:,:,0], shift=shift, sigma=np.random.uniform(0,0.2,1))[None,:])
            aug_labels.append(im_out[:,:,1][None,:]) 
            aug_imgs.append(add_gamma(im_out[:,:,0], gamma=0.3)[None,:]) # random gamma enhancement. 
            aug_labels.append(im_out[:,:,1][None,:])
            
    return np.concatenate(aug_imgs, axis=0), np.concatenate(aug_labels, axis=0)
    
    
def apply_elastic_transform_intensity_multi(imgs, labels, strength=0.08, shift=0.3, N=20):
    
    from skimage.exposure import rescale_intensity
    aug_imgs = []
    aug_labels = []
    n_imgs = len(imgs)    

    for i in range(n_imgs):
        im = rescale_intensity(imgs[i])
        lab = labels[i]

        im_ = np.dstack([im, lab])
#        print im_.shape
        for j in range(N):
            im_out = elastic_transform(im_, im_.shape[1] * strength, im_.shape[1] * strength, im_.shape[1] * strength, random_state=None)

            aug_imgs.append(im_out[:,:,0][None,:]) # no noise
            aug_labels.append(im_out[:,:,1:][None,:])  # with noise.             
            aug_imgs.append(add_noise(im_out[:,:,0], shift=shift, sigma=np.random.uniform(0,0.2,1))[None,:])
            aug_labels.append(im_out[:,:,1:][None,:]) 
            aug_imgs.append(add_gamma(im_out[:,:,0], gamma=0.3)[None,:]) # random gamma enhancement. 
            aug_labels.append(im_out[:,:,1:][None,:])
            
    return np.concatenate(aug_imgs, axis=0), np.concatenate(aug_labels, axis=0)
    
    
def find_circular_stuff(binary, ecc_thresh=0.8):
    from skimage.measure import label, regionprops
    from skimage.morphology import skeletonize
    
#    mask = skeletonize(binary)
    mask = binary.copy()
    labelled = label(mask)
    
    reg = regionprops(labelled)
    
    eccs = []
    for re in reg:
        eccs.append(re.eccentricity)
    eccs = np.hstack(eccs)
    
    uniq_labels = np.unique(labelled)[1:]
    good = uniq_labels[eccs <= ecc_thresh]
    
    new_mask = np.zeros_like(labelled)
    
    for g in good:
        new_mask[labelled==g] = g

    return new_mask, eccs
    
    
def filter_segmentations(binary, ecc_thresh=0.8):
    
    from  skimage.morphology import square, binary_closing
#     apply circularity cutoffs.... and size cutoffs. 
    binary = binary_closing(binary, square(1))
    
    return find_circular_stuff(binary, ecc_thresh)
    

def apply_gaussian_to_dots(img, sigma, thresh=0):
    
    from skimage.measure import label, regionprops
    from skimage.filters import gaussian
    
    binary = img > thresh
    labelled = label(binary)
    regions = regionprops(labelled)
    
    blank = np.zeros_like(binary)
    for reg in regions:
        y,x = reg.centroid
        y = int(y)
        x = int(x)
        blank[y,x] = 1
        
    blank = gaussian(blank, sigma=sigma, mode='reflect')
    
    return blank 
    
    
#def create_dot_annotations(xstack, ystack, sigma=5):
#    
#    from skimage.filters import threshold_otsu, gaussian
#    from skimage.measure import label, regionprops
#
#    y_ = []
#    x_ = []
#    
#    for i in range(len(ystack)):
#        im = ystack[i]
#        dapi = xstack[i]
#
#        n_channels = im.shape[-1]
#        im_out = []
#
#        for j in range(n_channels):
#            if len(np.unique(im[:,:,j])) > 1:
#                out = apply_gaussian_to_dots(img, sigma, thresh=0)
##            thresh = 0
#            binary = im>0
#            labelled = label(binary)
#            regions = regionprops(labelled)
#            
#            blank = np.zeros_like(binary)
#            for reg in regions:
#                y,x = reg.centroid
#                y = int(y)
#                x = int(x)
#                blank[y,x] = 1
#                
#            blank = gaussian(blank, sigma=sigma, mode='reflect')
#    #        blank = (len(np.unique(labelled))-1)*blank / float(np.sum(blank)) # make sure sum to the correct number... 
#    
#            y_.append(blank[None,:])
#            x_.append(dapi[None,:])
#        
#    y_ = np.concatenate(y_, axis=0)
#    x_ = np.concatenate(x_, axis=0)
#    
#    return x_, y_
    
    
def create_dot_annotations_multi(xstack, ystack, sigma=5, thresh=0):
    
    from skimage.filters import threshold_otsu, gaussian
    from skimage.measure import label, regionprops

    y_ = []
    x_ = []
    
    for i in range(len(ystack)):
        im = ystack[i]
        dapi = xstack[i]

        n_channels = im.shape[-1]
        im_out = []

        for j in range(n_channels):
            if len(np.unique(im[:,:,j])) > 1:
                out = apply_gaussian_to_dots(im[:,:,j], sigma, thresh=thresh)
                im_out.append(out)

        if len(im_out) == n_channels:
            im_out = np.dstack(im_out)
            y_.append(im_out[None,:])
            x_.append(dapi[None,:])
        
    y_ = np.concatenate(y_, axis=0)
    x_ = np.concatenate(x_, axis=0)
    
    return x_, y_
    
    
def read_multiimg_PIL(tiffile):
    
    """
    Use pillow library to read .tif/.TIF files. (single frame)
    
    Input:
    ------
    tiffile: input .tif file to read, can be multipage .tif (string)
    frame: desired frarme number given as C-style 0-indexing (int)

    Output:
    -------
    a numpy array that is either:
        (n_frames x n_rows x n_cols) for grayscale or 
        (n_frames x n_rows x n_cols x 3) for RGB

    """

    from PIL import Image
    import numpy as np

    img = Image.open(tiffile)

    imgs = []
    read = True

    frame = 0

    while read:
        try:
            img.seek(frame) # select this as the image
            imgs.append(np.array(img)[None,:,:])
            frame += 1
        except EOFError:
            # Not enough frames in img
            break

    return np.concatenate(imgs, axis=0)

def extract_dots(img, color):
    
    mask_r = img[:,:,:,0] == color[0]
    mask_g = img[:,:,:,1] == color[1]
    mask_b = img[:,:,:,2] == color[2]
    
    mask = np.logical_and(mask_r, np.logical_and(mask_g, mask_b))
    
    dots = img[:,:,:,2]*mask
    
    return dots 
    
    
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
    
        
def annotations_to_dots(xstack, ystack):
    
    from skimage.measure import label, regionprops
    from skimage.filters import gaussian
    
    cells = []
    dots = []

    for i in range(len(ystack)):
        y = ystack[i]
        labelled = label(y>10)
        n_regions = len(np.unique(labelled)) - 1
        if n_regions == 2:
            # retrieve centroids of labelled regions. 
            cents = ret_centroids_regionprops(labelled)
        else:
            # it should be 1 and we use local peaks to retrieve.
            cents = ret_centroids_localpeaks(y>10)
            
        # check if centroids == 2. 
        if len(cents) == 2:
            cells.append(xstack[i][None,:])
            new_y = np.zeros(y.shape, dtype=np.int)
            cents = cents.astype(np.int)
            for cent in cents:
                new_y[cent[0], cent[1]] = 1

            dots.append(new_y[None,:])
            
    return np.concatenate(cells, axis=0), np.concatenate(dots, axis=0)
    
    
def annotations_to_dots_multi(xstack, ystack):
    
    from skimage.measure import label, regionprops
    from skimage.filters import gaussian
    
    cells = []
    dots = []

    for i in range(len(ystack)):
        y = ystack[i]
        n_rows, n_cols, n_channels = y.shape
        y_out = []

        for j in range(n_channels):
            labelled = label(y[:,:,j]>10)
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
            elif j > 0:
                cents = ret_centroids_regionprops(labelled)
                new_y = np.zeros((n_rows, n_cols), dtype=np.int)
                cents = cents.astype(np.int)
                if len(cents) == 1:
                    new_y[cents[:,0], cents[:,1]] = 1
                    y_out.append(new_y)

        if len(y_out) == n_channels:
            cells.append(xstack[i][None,:])
            dots.append(np.dstack(y_out)[None,:])
            
    return np.concatenate(cells, axis=0), np.concatenate(dots, axis=0)
    
    
def rescale_intensity_stack(xstack):
    
    from skimage.exposure import rescale_intensity
    
    xx = np.concatenate([rescale_intensity(im)[None,:] for im in xstack], axis=0)
    
    return xx



def compile_train_test_data(in_files, typ='single', which=0):
    
    import scipy.io as spio
    
    if typ=='single':
        f = spio.loadmat(in_files[which])
        X_train = f['X_train']; Y_train = f['Y_train']; D_train = f['D_train']
        X_test = f['X_test']; Y_test=f['Y_test']; D_test=f['D_test']
        
        return (X_train,Y_train,D_train), (X_test,Y_test,D_test)
    
    else:
        
        X_train_all = []
        Y_train_all = []
        D_train_all = []
        
        X_test_all = []
        Y_test_all = []
        D_test_all = []
        
        for f in in_files:
            f = spio.loadmat(f)
            X_train = f['X_train']; Y_train = f['Y_train']; D_train = f['D_train']
            X_test = f['X_test']; Y_test=f['Y_test']; D_test=f['D_test']
            
            X_train_all.append(X_train)
            Y_train_all.append(Y_train)
            D_train_all.append(D_train)
            
            X_test_all.append(X_test)
            Y_test_all.append(Y_test)
            D_test_all.append(D_test)
        
        X_train_all = np.concatenate(X_train_all, axis=0)
        Y_train_all = np.concatenate(Y_train_all, axis=0)
        D_train_all = np.hstack(D_train_all)
        
        X_test_all = np.concatenate(X_test_all, axis=0)
        Y_test_all = np.concatenate(Y_test_all, axis=0)
        D_test_all = np.hstack(D_test_all)
        
        return (X_train_all,Y_train_all,D_train_all), (X_test_all,Y_test_all,D_test_all)


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
#   Load the train-test_data
#==============================================================================
    
    in_files = ['Training_Testing_patches_sectioned-Early.mat',
                'Training_Testing_patches_sectioned-Mid.mat',
                'Training_Testing_patches_sectioned-Late.mat']

###==============================================================================
###   Distort samples .. 
###==============================================================================
    stage = 'Early'
#    # Load up the single model first. 
#    (X_train,Y_train,D_train), (X_test,Y_test,D_test) = compile_train_test_data(in_files, typ='single', which=0)
#    
#    X_train = X_train / 255.
#    
####==============================================================================
####   Apply Deformations
####==============================================================================
#    train_x, train_y = apply_elastic_transform_intensity_multi(X_train/255., Y_train, strength=0.2, shift=0, N=50) # can try to use strength lower. 
#    train_x, train_y_dots = create_dot_annotations_multi(train_x, train_y, sigma=2, thresh=0) # turn into dots. 
#    
#    test_x, test_y_dots = create_dot_annotations_multi(X_test/255., Y_test, sigma=2, thresh=0) # turn into dots. 
#    
##     do we deform also ? -> hm..... no?? 
##    X_test = X_test / 255.
##    test_x, test_y_dots = create_dot_annotations_multi(X_test, Y_test, sigma=2, thresh=0)
##    test_x = rescale_intensity_stack(test_x)
#    
#    
#    # try saving the data
#    spio.savemat('pooled_augmented_train-test-compress-%s.mat' %(stage), {'X_train':train_x,
#                                                     'Y_train_dots':train_y_dots, 
#                                                     'X_test':test_x,
#                                                     'Y_test_dots':test_y_dots}, do_compression=True)
    
    
    
    # too noisy for validaton -> augment the test data too !.
#    test_x, test_y = apply_elastic_transform_intensity_multi(X_test/255., Y_test, strength=0.2, shift=0, N=50) # can try to use strength lower. 
#    test_x, test_y_dots = create_dot_annotations_multi(test_x, test_y, sigma=2, thresh=0) # turn into dots. 
#    
#    
#    train_x = train_x[:,:,:,None]
#    train_y_dots = train_y_dots[:,:,:,None]
#    test_x = test_x[:,:,:,None]
#    test_y = test_y_dots[:,:,:,None]
    train_test_data = spio.loadmat('pooled_augmented_train-test-compress-%s.mat' %(stage))
    
    train_x = train_test_data['X_train']
    train_y_dots = train_test_data['Y_train_dots']
    test_x = train_test_data['X_test']
    test_y_dots = train_test_data['Y_test_dots']
    
    train_x = train_x[:,:,:,None]
    train_y_dots = train_y_dots[:,:,:,None]
    test_x = test_x[:,:,:,None]
    test_y = test_y_dots[:,:,:,None]
    
    #==============================================================================
    #    U-net Model Construction.  
    #==============================================================================
    # save out and construct a FCNN....(probably need a scale transform)
    ISZ = None
    N_Cls = 2
    
    inputs = Input((ISZ, ISZ, 1))
    conv1 = Conv2D(32, (3, 3), activation='selu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='selu', padding='same')(conv1)
    
#    # replace to get batch normalization 
#    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
#    conv1 = BatchNormalization()(conv1)
#    conv1 = Activation('selu')(conv1)
#    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
#    conv1 = BatchNormalization()(conv1)
#    conv1 = Activation('selu')(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='selu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='selu', padding='same')(conv2)
    
#    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
#    conv2 = BatchNormalization()(conv2)
#    conv2 = Activation('selu')(conv2)
#    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
#    conv2 = BatchNormalization()(conv2)
#    conv2 = Activation('selu')(conv2)
    
#    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
##    conv3 = Conv2D(128, (3, 3), activation='selu', padding='same')(pool2)
##    conv3 = Conv2D(128, (3, 3), activation='selu', padding='same')(conv3)
#    
##    conv3 = Conv2D(64, (3, 3), padding='same')(pool2)
##    conv3 = BatchNormalization()(conv3)
##    conv3 = Activation('relu')(conv3)
##    conv3 = Conv2D(64, (3, 3), padding='same')(conv3)
##    conv3 = BatchNormalization()(conv3)
##    conv3 = Activation('relu')(conv3)
#    
#    conv3 = Conv2D(128, (3, 3), activation='selu', padding='same')(pool2)
#    conv3 = Conv2D(128, (3, 3), activation='selu', padding='same')(conv3)
#    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
##
##    conv4 = Conv2D(72, (3, 3), activation='relu', padding='same')(pool3)
##    conv4 = Conv2D(72, (3, 3), activation='relu', padding='same')(conv4)
##    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
##    
##    conv5 = Conv2D(72, (3, 3), activation='relu', padding='same')(pool4)
##    conv5 = Conv2D(72, (3, 3), activation='relu', padding='same')(conv5)
##    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
###
##    conv6 = Conv2D(90, (3, 3), activation='relu', padding='same')(pool5)
##    conv6 = Conv2D(90, (3, 3), activation='relu', padding='same')(conv6)
##
##    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv5], mode='concat', concat_axis=-1)
##    conv7 = Conv2D(72, (3, 3), activation='relu', padding ='same')(up7)
##    conv7 = Conv2D(72, (3, 3), activation='relu', padding ='same')(conv7)
##
#    up8 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=-1)
#    up8 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv3), conv2])
#    conv8 = Conv2D(64, (3, 3), activation='selu', padding='same')(up8)
#    conv8 = Conv2D(64, (3, 3), activation='selu', padding='same')(conv8)
#
#    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv3], mode='concat', concat_axis=-1)
#    up9 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv2), conv1])
#    conv9 = Conv2D(32, (3, 3), activation='selu', padding='same')(up9)
#    conv9 = Conv2D(32, (3, 3), activation='selu', padding='same')(conv9)
#
    up10 = merge([UpSampling2D(size=(2, 2))(conv2), conv1], mode='concat', concat_axis=-1)
    conv10 = Conv2D(32, (3, 3), activation='selu', padding='same')(up10)
    conv10 = Conv2D(32, (3, 3), activation='selu', padding='same')(conv10)
#    
#    up11 = merge([UpSampling2D(size=(2, 2))(conv10), conv1], mode='concat', concat_axis=-1)
#    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(up11)
#    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv11)
#    conv10 = merge([conv9, inputs], mode='concat', concat_axis=-1)
#    up10 = merge([inputs, conv9], mode='concat', concat_axis=-1) # also feed in the original image as well as the upsampled. 
    conv12 = Conv2D(N_Cls, (1, 1), activation='relu')(conv10)

    model = Model(inputs=inputs, outputs=conv12)
    model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy']) # avoid over detection ?

    model.summary()

    
    #==============================================================================
    #   Model Fitting with Early Stopping.   
    #==============================================================================
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    

    train_y_dots = train_y_dots* 1000.
    test_y_dots = test_y_dots * 1000.
    
    batch_size = 128
    epochs = 200
    
#        """ Set some early stopping parameters """
    early_stop = EarlyStopping(monitor='val_loss', 
                               min_delta=0.001, 
                               patience=15, 
                               mode='min', 
                               verbose=1)
    
    checkpoint = ModelCheckpoint('Multi-S1_32x32_selu_all_sigma2_mse-%s-notestaug' %(stage), 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min', 
                                 period=1)

    # single model
#    history = model.fit(train_x, train_y_dots[:,:,:,0,0][:,:,:,None],
#              batch_size=batch_size,
#              epochs=epochs,
#              validation_data=(test_x, test_y_dots[:,:,:,0][:,:,:,None]), shuffle=True, 
#              callbacks = [early_stop, checkpoint])
    
    history = model.fit(train_x, train_y_dots[:,:,:,0],
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(test_x, test_y_dots), shuffle=True, 
              callbacks = [early_stop, checkpoint])
    
    plt.figure()
    plt.plot(history.history['loss'], 'r')
    plt.plot(history.history['val_loss'], 'g')
#        
#####        
######### =============================================================================
#########   test
######### =============================================================================
#######
###    
    for i in range(len(test_x))[:10]:
        
#        imm = X_test[i]/255.
        imm = np.squeeze(test_x[i])
        imm = rescale_intensity(imm)
#        imm = equalize_adapthist((imm/255.)[:,:,0], clip_limit=0.01)[:,:,None]
#        imm = imm /255.
        
        out = model.predict(imm[None,:,:,None])/1000.
        
#        plt.figure()
#        plt.subplot(121)
#        plt.imshow(np.squeeze(imm), cmap='gray')
#        plt.subplot(122)
#        plt.imshow(np.squeeze(out))
#        plt.show()
        
        plt.figure()
        plt.subplot(131)
        plt.imshow(np.squeeze(imm), cmap='gray')
        plt.subplot(132)
        plt.imshow(np.squeeze(out[:,:,:,0]))
        plt.subplot(133)
        plt.imshow(np.squeeze(out[:,:,:,1]))
#        plt.subplot(154)
#        plt.imshow(np.squeeze(out))
#        plt.subplot(155)
#        plt.imshow(Y_test[i][:,:,1])
#        plt.show()
        
        print np.sum(np.squeeze(out))
#    
#    
#    
#    
    

        
        
        
        
        
    
    