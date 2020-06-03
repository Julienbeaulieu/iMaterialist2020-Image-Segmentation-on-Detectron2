import numpy as np
from PIL import Image
import math

# Rle helper functions

def rle_decode_string(rle, h, w):
    '''
    rle: run-length encoded image mask, as string
    h: heigh of image on which RLE was produced
    w: width of image on which RLE was produced
    returns a binary mask with the same shape
    '''
    mask = np.full(h * w, 0, dtype=np.uint8)
    annotation = [int(x) for x in rle.split(' ')]
    for i, start_pixel in enumerate(annotation[::2]):
        mask[start_pixel: start_pixel + annotation[2 * i + 1]] = 1
    mask = mask.reshape((h, w), order='F')

    return mask


def mask_to_KaggleRLE_old(img):
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

def mask_to_KaggleRLENew(img):
    pixels = img.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return ' '.join(str(x) for x in rle)



def mask_to_KaggleRLE_downscale(img, max_size=1024):
    '''
    Adaptive funciton to first DOWNSCALE the image before running RLE
    # Source: https://stackoverflow.com/a/28453021
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    # img is a tensor
    img_array = np.array(img)
    pil_image = Image.fromarray(img_array)
    width_current = pil_image.size[0]
    height_current = pil_image.size[1]

    longest_edge = max(width_current, height_current)

    # Always rescale
    if (width_current > height_current):
        new_width = max_size
        scaled_height = max_size / float(width_current) * height_current
        # Must floor the mask
        new_height = int(math.floor(scaled_height))
    else:
        scale_width = max_size / float(height_current) * width_current
        # Must floor the mask
        new_width = int(math.floor(scale_width))
        new_height = max_size

    pil_image = pil_image.resize((new_width, new_height), Image.NEAREST)

    image_array = np.array(pil_image)
    return mask_to_KaggleRLENew(image_array)


def rle_encode(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle


def rle2bbox(rle, shape):
    '''
    Get a bbox from a mask which is required for Detectron 2 dataset
    rle: run-length encoded image mask, as string
    shape: (height, width) of image on which RLE was produced
    Returns (x0, y0, x1, y1) tuple describing the bounding box of the rle mask

    Note on image vs np.array dimensions:

        np.array implies the `[y, x]` indexing order in terms of image dimensions,
        so the variable on `shape[0]` is `y`, and the variable on the `shape[1]` is `x`,
        hence the result would be correct (x0,y0,x1,y1) in terms of image dimensions
        for RLE-encoded indices of np.array (which are produced by widely used kernels
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
        raise ValueError("invalid RLE or image dimensions: x1=%d > shape[1]=%d" % (
            x1, shape[1]
        ))

    return x0, y0, x1, y1

def rle_decode_new(rle_str, mask_shape):
    # This is the reverse decoding mechanism
    # Source: https://www.kaggle.com/stainsby/fast-tested-rle-and-input-routines
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(mask_shape[::-1]).T

def refine_masks(masks):
    """
    Take a series of masks
    sort them from small to large (yet preserve order information)
    Add each one to union, keep track of already counted pixel coordinate positions to make the masks "uniquefy"
    Convert each "uniquefied" mask to RLE
    return them in the order originally found

    :param masks:
    :param n_labels:
    :return:
    """
    # Source: https://github.com/abhishekkrthakur/imat-fashion/blob/master/predict.py
    n_labels = len(masks)

    # Early return if nothing.
    if n_labels == 0:
        # Return empty string
        return []

    masks = np.array(masks)

    # Compute the areas of each mask
    #mask_areas = np.sum(masks.reshape(-1, masks.shape[0]), axis=1)
    masks_areas = [np.sum(np.sum(mask)) for mask in masks]

    # Preserve the original order?
    masks_areas_ordered = list(enumerate(masks_areas))
    masks_areas_ordered_sorted = sorted(masks_areas_ordered, key=lambda a: a[1])

    # One reference mask is created to be incrementally populated
    # This has same as number of labels.

    # Generate a blank union mask, which all labels will be iteratively updating to ensure no overlap pixels.
    union_mask = np.zeros(masks[0].shape, dtype=bool)

    # Refined masks
    uniquefied_mask = []

    # Iterate from the smallest, so smallest ones are preserved
    # Second parameter is area, useless.
    for mask_index, _ in masks_areas_ordered_sorted:
        # Current Mask:
        mask_current = masks[mask_index, :, :]

        # unionized version fo the current mask: Default to false/0 if not defined.
        # not logical_not, it turns all False (default) to True. True&True = True
        # All true for the first iteration only.
        union_mask_inverted = np.logical_not(union_mask)

        uniquefied_mask_current = np.logical_and(
            mask_current,
            union_mask_inverted # not logical_not, it turns all False (default) to True. True&True = True
        )

        uniquefied_mask.append((mask_index, uniquefied_mask_current))

        # update the union mask to include the latest calculation
        union_mask = np.logical_or(mask_current, union_mask)

    # sort this by original index.
    uniquefied_mask.sort(key=lambda a: a[0])

    refined_rle = []

    # Iterate through masks last axis
    for mask_index, uniquefied_mask_current in uniquefied_mask:

        # Change this line to determine whether to use downscaled KaggleRLE or regular KaggleRLE conversion process
        rle = mask_to_KaggleRLE_downscale(uniquefied_mask_current)
        #rle = mask_to_KaggleRLE_old(uniquefied_mask_current)
        #rle = mask_to_KaggleRLE_downscale(uniquefied_mask_current)
        refined_rle.append(rle)
        # Sanity check on uniquefying reduction
        # print(f"Original: {masks_areas[mask_index]}, {np.sum(np.sum(uniquefied_mask_current))}")

    return refined_rle
