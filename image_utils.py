# -*- coding: utf-8 -*-
import numpy as np
import scipy

from numpy import min, mgrid, exp

import math
import cv2
import matplotlib.pyplot as plt

from PIL import Image
import skimage


def otf2psf(otf, outsize=None):
    insize = np.array(otf.shape)
    psf = np.fft.ifftn(otf, axes=(0, 1))

    for axis, axis_size in enumerate(insize):
        psf = np.roll(psf, np.floor(axis_size / 2).astype(int), axis=axis)

    if type(outsize) != type(None):
        insize = np.array(otf.shape)
        outsize = np.array(outsize)
        n = max(np.size(outsize), np.size(insize))
        colvec_out = outsize.flatten().reshape((np.size(outsize), 1))
        colvec_in = insize.flatten().reshape((np.size(insize), 1))
        outsize = np.pad(colvec_out,
                         ((0, max(0, n - np.size(colvec_out))), (0, 0)),
                         mode="constant")
        insize = np.pad(colvec_in,
                        ((0, max(0, n - np.size(colvec_in))), (0, 0)),
                        mode="constant")

        pad = (insize - outsize) / 2

        if np.any(pad < 0):
            print(
                "otf2psf error: OUTSIZE must be smaller than or equal than OTF size"
            )

        prepad = np.floor(pad)
        postpad = np.ceil(pad)
        dims_start = prepad.astype(int)
        dims_end = (insize - postpad).astype(int)

        for i in range(len(dims_start.shape)):
            psf = np.take(psf, range(dims_start[i][0], dims_end[i][0]), axis=i)

    n_ops = np.sum(otf.size * np.log2(otf.shape))
    psf = np.real_if_close(psf, tol=n_ops)
    return psf


def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)
    if np.alltrue(imshape == shape):
        return image
    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")
    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")
    pad_img = np.zeros(shape, dtype=image.dtype)
    idx, idy = np.indices(imshape)
    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)
    pad_img[idx + offx, idy + offy] = image
    return pad_img


# https://github.com/aboucaud/pypher/blob/master/pypher/pypher.py
def psf2otf(psf, shape=None):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if type(shape) == type(None):
        shape = psf.shape

    shape = np.array(shape)

    if np.all(psf == 0):
        return np.zeros(shape)

    if len(psf.shape) == 1:
        psf = psf.reshape((1, psf.shape[0]))

    inshape = psf.shape
    psf = zero_pad(psf, shape, position='corner')

    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf, axes=(0, 1))
    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)
    return otf


def fspecial_average(hsize=3):
    """Smoothing filter"""
    return np.ones((hsize, hsize)) / hsize**2


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1),
                         np.arange(-siz[0], siz[0] + 1))
    arg = -(x * x + y * y) / (2 * std * std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


# Return the truncated Gaussian
# corresponding to fspecial('gaussian', sz, sd)
# (Another implementation for fspecial_gaussian)
def fGaussian(sz, sd):
    m, n = sz
    h, k = m // 2, n // 2
    x, y = np.meshgrid(np.linspace(-h, h, m), np.linspace(-k, k, n))
    d = np.sqrt(x * x + y * y)
    g = (1.0 / (2.0 * np.pi * sd)) * np.exp(-(d**2) / (2.0 * sd**2))
    gg = np.zeros((g.shape[0], g.shape[1], 1))
    gg[:, :, 0] = g
    return gg


def fspecial_laplacian(alpha):
    alpha = max([0, min([alpha, 1])])
    h1 = alpha / (alpha + 1)
    h2 = (1 - alpha) / (alpha + 1)
    h = [[h1, h2, h1], [h2, -4 / (alpha + 1), h2], [h1, h2, h1]]
    h = np.array(h)
    return h


def fspecial_gauss(size, sigma):
    x, y = mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


# Level-set curvature.
# Approximate the derivatives using Sobel filter
def curv_Sobel(img):
    a = 0.001
    grads_x = scipy.ndimage.sobel(img, axis=0)
    grads_y = scipy.ndimage.sobel(img, axis=1)
    igrad2 = grads_x * grads_x + grads_y * grads_y
    igrad1 = np.sqrt(igrad2)
    ngrads_x = grads_x / (igrad1 + a)
    ngrads_y = grads_y / (igrad1 + a)
    icurv1x = scipy.ndimage.sobel(ngrads_x, axis=0)
    icurv1y = scipy.ndimage.sobel(ngrads_y, axis=1)
    icurv = icurv1x + icurv1y
    return icurv


# Level-set curvature.
# Approximate the derivatives using central difference (similar to the Matlab version)
def curv(img, curv_a=0.001):
    dx = np.array([[-1.0 / 2.0, 0.0, 1.0 / 2.0]])
    dy = np.array([[-1.0 / 2.0], [0.0], [1.0 / 2.0]])

    grads_x = np.zeros_like(img)
    grads_y = np.zeros_like(img)

    # Note:
    # correlate(A, B, mode='nearest') corresponds to imfilter(A, B, 'replicate') in Matlab
    if (len(img.shape) == 3):
        # Color images
        grads_x[:, :, 0] = scipy.ndimage.correlate(img[:, :, 0],
                                                   dx,
                                                   mode='nearest')
        grads_x[:, :, 1] = scipy.ndimage.correlate(img[:, :, 1],
                                                   dx,
                                                   mode='nearest')
        grads_x[:, :, 2] = scipy.ndimage.correlate(img[:, :, 2],
                                                   dx,
                                                   mode='nearest')
        grads_y[:, :, 0] = scipy.ndimage.correlate(img[:, :, 0],
                                                   dy,
                                                   mode='nearest')
        grads_y[:, :, 1] = scipy.ndimage.correlate(img[:, :, 1],
                                                   dy,
                                                   mode='nearest')
        grads_y[:, :, 2] = scipy.ndimage.correlate(img[:, :, 2],
                                                   dy,
                                                   mode='nearest')
    else:
        # Black and white
        grads_x = scipy.ndimage.correlate(img, dx, mode='nearest')
        grads_y = scipy.ndimage.correlate(img, dy, mode='nearest')

    igrad2 = grads_x * grads_x + grads_y * grads_y
    igrad1 = np.sqrt(igrad2)

    ngrads_x = grads_x / (igrad1 + curv_a)
    ngrads_y = grads_y / (igrad1 + curv_a)

    icurv1x = np.zeros_like(img)
    icurv1y = np.zeros_like(img)

    if (len(img.shape) == 3):
        icurv1x[:, :, 0] = scipy.ndimage.correlate(ngrads_x[:, :, 0],
                                                   dx,
                                                   mode='nearest')
        icurv1x[:, :, 1] = scipy.ndimage.correlate(ngrads_x[:, :, 1],
                                                   dx,
                                                   mode='nearest')
        icurv1x[:, :, 2] = scipy.ndimage.correlate(ngrads_x[:, :, 2],
                                                   dx,
                                                   mode='nearest')
        icurv1y[:, :, 0] = scipy.ndimage.correlate(ngrads_y[:, :, 0],
                                                   dy,
                                                   mode='nearest')
        icurv1y[:, :, 1] = scipy.ndimage.correlate(ngrads_y[:, :, 1],
                                                   dy,
                                                   mode='nearest')
        icurv1y[:, :, 2] = scipy.ndimage.correlate(ngrads_y[:, :, 2],
                                                   dy,
                                                   mode='nearest')
    else:
        icurv1x = scipy.ndimage.correlate(ngrads_x, dx, mode='nearest')
        icurv1y = scipy.ndimage.correlate(ngrads_y, dy, mode='nearest')

    icurv = icurv1x + icurv1y
    return icurv


# Butterworth filter
def lbutter(im, d, n):
    height, width = im.shape[0], im.shape[1]
    x = np.linspace(-math.floor(width / 2), math.floor((width - 1) / 2), width)
    y = np.linspace(-math.floor(height / 2), math.floor((height - 1) / 2),
                    height)
    xv, yv = np.meshgrid(x, y)
    out = 1 / (1 + (np.sqrt(2.0) - 1.0) * ((xv**2 + yv**2) / d**2)**n)
    return out


def imshow(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


# Read the image as an np array of floats in [0, 1]
# RGB
def imread(path):
    img = Image.open(path)
    f = np.asarray(img)
    f = skimage.img_as_float(f)


def imread_bgr(path):
    # Return: Numpy float32, HWC, BGR, [0,1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_GRAYSCALE
    img = img.astype(np.float32) / 255.

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def imread_uint(path, n_channels=3):
    # Return: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


def uint2single(img):
    return np.float32(img / 255.)


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.).round())


def uint162single(img):
    return np.float32(img / 65535.)


def single2uint16(img):
    return np.uint8((img.clip(0, 1) * 65535.).round())


def rgb2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img.astype(np.float32)

    if in_img_type != np.uint8:
        img *= 255.

    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img,
                        [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                         [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]

    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.

    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    in_img_type = img.dtype
    img.astype(np.float32)

    if in_img_type != np.uint8:
        img *= 255.

    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621],
                          [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [
                              -222.921, 135.576, -276.836
                          ]
    rlt = np.clip(rlt, 0, 255)

    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.

    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img.astype(np.float32)

    if in_img_type != np.uint8:
        img *= 255.

    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img,
                        [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                         [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]

    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.

    return rlt.astype(in_img_type)


def psnr(img1, img2, border=0):
    if not img1.shape == img2.shape:
        raise ValueError('Dimensions do not match.')

    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)

    if mse == 0:
        return float('inf')

    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2, border=0):
    if not img1.shape == img2.shape:
        raise ValueError('Dimensions do not match.')

    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    if img1.ndim == 2:
        return ssim_(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim_(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim_(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Invalid image dimensions.')


def ssim_(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
