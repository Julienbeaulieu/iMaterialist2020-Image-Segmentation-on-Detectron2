import numpy as np
from pycocotools.mask import encode, toBbox
from typing import List
from itertools import groupby


def mask_to_uncompressed_CocoRLE(binary_mask):
    """
    Source: https://stackoverflow.com/questions/49494337/encode-numpy-array-using-uncompressed-rle-for-coco-dataset
    """
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def mask_to_KaggleRLE(img):
    '''
    Source: https://www.kaggle.com/lifa08/run-length-encode-and-decode
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def KaggleRLE_to_mask1(mask_rle, shape):
    '''
    # Source: https://www.kaggle.com/lifa08/run-length-encode-and-decode
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return, Height Width Format
    Returns numpy array, 1 - mask, 0 - background

    (in fortran format?)

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def KaggleRLE_to_CocoRLE(KaggleRLE: str, h: int, w: int) -> List[dict]:
    """
    This wrapper function converts kaggle KaggleRLE to binary, then convert that binary to COCORLE.
    :param KaggleRLE:
    :param h:
    :param w:
    :return:
    """
    # Conver to binary using tried and true masks.
    mask = KaggleRLE_to_mask(KaggleRLE, h, w)

    # using PyCoCoAPI to convert binary mask to CocoRLE format, which are re a LIST of dict of run-length encoding of binary masks.
    CocoRLE = encode(np.asfortranarray(mask))

    return CocoRLE

def KaggleRLE_to_CocoBoundBoxes(KaggleRLE: str, h: int, w: int) -> List[int]:

    # Generate the KaggleRLE in coco format
    CocoRLE = KaggleRLE_to_CocoRLE(KaggleRLE, h, w)

    # Generate the BBS using the CocoAPI.
    CocoBBS = toBbox(CocoRLE)

    return CocoBBS

def KaggleRLE_to_mask(rle, h, w):
    '''
    rle: run-length encoded image mask, as string
    h: heigh of image on which KaggleRLE was produced
    w: width of image on which KaggleRLE was produced

    returns a binary mask with the same shape

    '''
    mask = np.full(h * w, 0, dtype=np.uint8)
    annotation = [int(x) for x in rle.split(' ')]
    for i, start_pixel in enumerate(annotation[::2]):
        mask[start_pixel: start_pixel + annotation[2 * i + 1]] = 1
    mask = mask.reshape((h, w), order='F')

    return mask


def KaggleRLE_to_bbox(rle, shape):
    '''
    Get a bbox from a mask which is required for Detectron 2 dataset
    rle: run-length encoded image mask, as string
    shape: (height, width) of image on which KaggleRLE was produced
    Returns (x0, y0, x1, y1) tuple describing the bounding box of the rle mask

    Note on image vs np.array dimensions:

        np.array implies the `[y, x]` indexing order in terms of image dimensions,
        so the variable on `shape[0]` is `y`, and the variable on the `shape[1]` is `x`,
        hence the result would be correct (x0,y0,x1,y1) in terms of image dimensions
        for KaggleRLE-encoded indices of np.array (which are produced by widely used kernels
        and are used in most kaggle competitions datasets)
    '''

    a = np.fromiter(rle.split(), dtype=np.uint)
    a = a.reshape((-1, 2))  # an array of (start, length) pairs
    a[:, 0] -= 1  # `start` is 1-indexed

    y0 = a[:, 0] % shape[0]
    y1 = y0 + a[:, 1]
    if np.any(y1 > shape[0]):
        # got `y` overrun, meaning that there are a pixels in mask on 0 and shape[0] position
        y0 = 0
        y1 = shape[0]
    else:
        y0 = np.min(y0)
        y1 = np.max(y1)

    x0 = a[:, 0] // shape[0]
    x1 = (a[:, 0] + a[:, 1]) // shape[0]
    x0 = np.min(x0)
    x1 = np.max(x1)

    if x1 > shape[1]:
        # just went out of the image dimensions
        raise ValueError("invalid KaggleRLE or image dimensions: x1=%d > shape[1]=%d" % (
            x1, shape[1]
        ))

    return x0, y0, x1, y1