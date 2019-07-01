import os.path
import random
import math

import numpy as np
import tifffile as tiff


DEFAULT_PATCH_SIZE = 128

"""
Returns a random augmented patch from the image

img:     numpy array of shape (x_size, y_size, nb_channels)
mask:    binary (one-hot) numpy array of shape (x_size, y_size, nb_classes)
size:    size of random patch (square)

returns: patch with shape(size, size, nb_channels) and its mask
"""
def get_rand_patch(img, mask, size=DEFAULT_PATCH_SIZE):
    assert len(img.shape) == 3     \
           and img.shape[0] > size \
           and img.shape[1] > size \
           and img.shape[0:2] == mask.shape[0:2]
    
    # SpaceNet images have 8 bands, we take only 3
    
    xs = random.randint(0, img.shape[0] - size)
    ys = random.randint(0, img.shape[1] - size)
    patch_img  = img[xs:xs+size, ys:ys+size]
    patch_mask = mask[xs:xs+size, ys:ys+size]
    
    # apply random transformations
    rt = np.random.randint(0, 7)
    if rt == 0:
        # horizontal flip
        patch_img  = patch_img[::-1, :, :]
        patch_mask = patch_mask[::-1, :, :]
    elif rt == 1:
        # vertical flip
        patch_img  = patch_img[:, ::-1, :]
        patch_mask = patch_mask[:, ::-1, :]
    elif rt == 2:
        # transpose
        patch_img = patch_img.transpose([1, 0, 2])
        patch_mask = patch_mask.transpose([1, 0, 2])
    elif rt == 3:
        # 90 degree rotation
        patch_img = np.rot90(patch_img, 1)
        patch_mask = np.rot90(patch_mask, 1)
    elif rt == 4:
        # 180 degree rotation
        patch_img = np.rot90(patch_img, 2)
        patch_mask = np.rot90(patch_mask, 2)
    elif rt == 5:
        # 270 degree rotation
        patch_img = np.rot90(patch_img, 3)
        patch_mask = np.rot90(patch_mask, 3)
    else:
        # no transformation
        pass
    
    return patch_img, patch_mask


"""
returns specified number of patches

x_dict:     (input) image dictionary (image_id -> image)
y_dict:     (output) mask dictionary (image_id -> image)
nb_pathces: number of patches to return
size:       size of patches

returns:    x and y, both numpy arrays of shape
            (nb_patches, patch_size, patch_size, nb_channels)
"""
def get_patches(x_dict, y_dict, nb_patches, size=DEFAULT_PATCH_SIZE):
    x = []
    y = []
    total_patches = 0
    
    while total_patches < nb_patches:
        img_id = random.sample(x_dict.keys(), 1)[0]
        img  = x_dict[img_id]
        mask = y_dict[img_id]
        img_patch, mask_patch = get_rand_patch(img, mask, size)
        x.append(img_patch)
        y.append(mask_patch)
        total_patches += 1
    
    print("Generated {} patches".format(total_patches))
    
    return np.array(x), np.array(y)