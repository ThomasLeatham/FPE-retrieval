# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 18:40:53 2021

@author: talea4
"""
import numpy as np
from skimage import restoration
import scipy.ndimage as ndimage
import skimage.transform as transform
def deconvolve(image,sigma,deconv_type='unsupervised_wiener',iterations=10):
    """Deconvolution of acquired images to correct for detector's PSF using 
    either Unsupervised Wiener or Richardson-Lucy deconvolution with  a given simga
    Args:
        image (numpy.ndarray): image to deconvolve
        sigma (float): standard deviation of the blurring
        deconv_type (str): 'unsupervised_wiener' or 'richardson_lucy'
        iteraions (float): number of iterations for Richardson-Lucy deconvolution
    Raises:
        Exception: 'Filter not found'
    Returns:
        restored_image (numpy.ndarray): image after deconvolution 
    """
    Nblur=int(np.floor(sigma*6))
    x,y=np.meshgrid(np.arange(0,Nblur),np.arange(0,Nblur))
    x=(x-((Nblur-1)/2))
    y=(y-((Nblur-1)/2))
    blur=np.exp(-(x**2+y**2)/sigma**2/2)
    blur=blur/np.sum(blur)
    MAX=np.max(image)
    #Restore image using unsupervised_wiener algorithm
    if deconv_type=='unsupervised_wiener':
        restored_image,_=restoration.unsupervised_wiener(image/MAX, blur,clip=False)
    #Restore image using Richardson-Lucy algorithm
    elif deconv_type=='richardson_lucy':
        restored_image=restoration.richardson_lucy(image/MAX, blur,iterations=iterations,clip=False)
    else:
        raise Exception('Filter not found')
    restored_image=restored_image*MAX
    return restored_image
def register_fourier(image1, image2, upsample_factor=1000,shift_only=False):
    """Wrapper for Fourier based image registration with sub-pixel accuracy
    from skimage.feature.register_translation

    Finds out translation to best match image2 to image1. The shifts
    returned are those that were used to match image2 to image1.

    florian.schaff@monash.edu

    Parameters
    ----------
    image1 : 2-d array
        target image, i.e. image2 is registered to image1
    image2 : 2-d array
        image that is transformed to fit image1
    upsample_factor : int
        upsampling of cross correlation peak to improve sub-pixel accuracy

    Returns
    ----------
    out : shifted image2, shifts

    """

    from skimage.registration import phase_cross_correlation
    shifts = phase_cross_correlation(image1, image2,upsample_factor=upsample_factor)[0]
    if shift_only==False:
        fim2 = np.fft.fftn(image2)
        image2_shifted = np.real(np.fft.ifftn(ndimage.fourier_shift(fim2, shifts)))
        return image2_shifted, shifts
    else:
        return shifts
def angle_from_2d_rotationmatrix(R):
    """computes the rotation angle from a given 2x2 rotation matrix

    florian.schaff@monash.edu

    Parameters
    ----------
    R : 2-d array
        2x2 rotation matrix

    Returns
    ----------
    out : float
        rotation angle in degree

    """
    if np.round(np.linalg.det(R), 6) == 1:
        return np.arctan2(R[0, 1], R[1, 1])*180/np.pi
    else:
        print('R is not a rotation matrix!')

def register_images(image1, image2, rotation=True, par_only=False,
                    maxcounts=500, epsilon=1e-8):
    """Wrapper for image registration using openCV for python (cv2). Based on

    https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/

    Finds out shift (and rotation if set to True) to best match image2 to
    image1. The shift and rotation angle returned are those that were applied
    to get from image1 to image2, i.e. NOT the ones needed to undo the
    transformation.

    florian.schaff@monash.edu

    Parameters
    ----------
    image1 : 2-d array
        target image, i.e. image2 is registered to image1
    image2 : 2-d array
        image that is transformed to fit image1
    rotation : boolean
        include rotation in the affine transformation, if False the algorithm
        is much faster
    par_only : boolean
        return only shift/angle parameters if True. Default: False
    maxcounts, epsilon : int, float
        termination criteria of the algorithm

    Returns
    ----------
    out : shifted image2, shift, angle [deg]
        shifted image2 as float32 unless par_only = True, then only shift and
        angle are returned

    Examples
    --------
    >>> image1 = np.zeros((200,200))
    >>> image1[50:150,50:150] = 1
    >>> image2 = np.roll(image1,(10,-5),axis=(0,1))
    >>> registered, shift, angle = register_images(image1,image2)

    """
    from cv2 import (MOTION_EUCLIDEAN, MOTION_TRANSLATION, findTransformECC,
                    TERM_CRITERIA_EPS, TERM_CRITERIA_COUNT, warpAffine,
                    INTER_LINEAR, WARP_INVERSE_MAP)

    if rotation:
        warp_mode = MOTION_EUCLIDEAN
    else:
        warp_mode = MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, maxcounts, epsilon)
    (cc, warp_matrix) = findTransformECC(image1.astype(np.float32),
                                       image2.astype(np.float32),
                                       warp_matrix, warp_mode,criteria,None,1)
    angle = angle_from_2d_rotationmatrix(warp_matrix[:, :2])
    translation = warp_matrix[::-1, 2]  # axes are swapped in cv
    if par_only:
        return translation, angle
    else:
        im_aligned = warpAffine(image2.astype(np.float32), warp_matrix,
                                image1.shape[::-1],  # axes are swapped in cv
                                flags=INTER_LINEAR + WARP_INVERSE_MAP)
        return im_aligned, translation, angle
def image_alignment(ref_image,mov_image,alignment_mode='fourier',rot=False):
    """Image registration to ensure that features in experimental images acuiqred at different
    distances line up. 
    Args:
        ref_image (numpy.ndarray): reference image that moving image should be aligned to
        mov_image (numpy.ndarray): moving image to be aligned to reference image
        alignment_mode (str): How to perofrm image resgistration, either in real or Fourier space. 
        Either 'fourier' or 'real'
        rot (bool): Extra paramter to specify if alignment_mode='real' (accounts for rotations)
    """
    if alignment_mode=='fourier':
        aligned_image,shifts=register_fourier(ref_image,mov_image,upsample_factor=1000,shift_only=False)
    elif alignment_mode=='real' and rot==False:
        aligned_image,shifts=register_images(ref_image,mov_image,rotation=False,par_only=False,maxcounts=500,epsilon=1e-8)
    else:
        aligned_image,shifts,rotations=register_images(ref_image,mov_image,rotation=True,par_only=False,maxcounts=500,epsilon=1e-8)
    return aligned_image
def image_alignment_list(image_list,list_align='next',alignment_mode='fourier',rot=False):
    """ Align PBI images using a specified method.
    Args:
        image_list (list of numpy.ndarray): PBI images to be algined
        list_align (str): string specifying how to perform alignment of images.
        Must be one of 'next', 'shortest' or 'longest'.
        'next' = align images to the previous image
        'shortest' = align all images to shortest distance image
        'longest' = align all images to longest distance image
        alignment_mode (str): see image_alignment 
        rot (bool): see image_alignment
    Returns:
        aligned_images (list of numpy.ndarray): aligned PBI images
    """
    aligned_images=[]
    if list_align=='next':
        first_align=image_alignment(image_list[0],image_list[1],alignment_mode=alignment_mode,rot=rot)
        aligned_images.append(first_align)
        for i in range(2,len(image_list)):
            aligned_image=image_alignment(aligned_images[i-2],image_list[i],alignment_mode=alignment_mode,rot=rot)
            aligned_images.append(aligned_image)
        aligned_images.insert(0,image_list[0])
    elif list_align=='shortest':
        for i in range(1,len(image_list)):
            aligned_image=image_alignment(image_list[0],image_list[i],alignment_mode=alignment_mode,rot=rot)
            aligned_images.append(aligned_image)
        aligned_images.insert(0,image_list[0])
    elif list_align=='longest':
        for i in range(len(image_list)-1):
            aligned_image=image_alignment(image_list[i],image_list[-1],alignment_mode=alignment_mode,rot=rot)
            aligned_images.append(aligned_image)
        aligned_images.insert(-1,image_list[-1])
    return aligned_images
def fix_mag(ssd,dist_list,pbi_list,dims,method='lowest',order=3,clip=False,anti_aliasing=True):
    """ Accounting for magnfication effects in PBI images prior to image alignment
    Args:
        ssd (float): source-to-sample distance in m
        dist_list (list of floats): list of propagation distances.
        Must be ordered from smallest to largest distance.
        pbi_list (list of numpy.ndarrays): list of experimental PBI images.
        Must be in the same order as dist_list. 
        dims (list of two ints): dimensions to crop all magnified images to.
        Specified as [H,W], i.e. image height then width.
        method (str): how to perform magnfication fix.
        lowest = magnify all images down to lowest value
        highest = magnify all images up to highest value
        order (int): interpolation used in skimage.transform.rescale.
        See skimage.transform for documentation
        clip (bool): whether to clip values in data range or not.
        See skimage.transform for documentation
        anti_aliasing (bool): whether to anti-aliase when down-sampling to avoid artefacts.
        Highly recommended. See skimage.transform for documentation
    Returns:
        result (list of numpy.ndarray): magnified/demagnified images 
        mag_factors (list of floats): geometric magnification factors associated
        to each propagation distance
        """
    mag_factors=[(dist+ssd)/ssd for dist in dist_list]
    mag_images=[]
    if method=='lowest':
        for i in range(0,len(mag_factors)):
            mag_img=transform.rescale(pbi_list[i],mag_factors[0]/mag_factors[i],order=order,clip=clip,anti_aliasing=anti_aliasing)
            mag_images.append(mag_img)
    else:
        H,W=pbi_list[0].shape
        for i in range(0,len(mag_factors)):
            mag_img=transform.rescale(pbi_list[i],mag_factors[i]/mag_factors[-1],order=order,clip=clip,anti_aliasing=anti_aliasing)
            mag_images.append(mag_img)
    result=[im[0:int(dims[0]),0:int(dims[1])] for im in mag_images]
    return result, mag_factors

        