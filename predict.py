import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

from patches import DEFAULT_PATCH_SIZE
from train   import (weights_path,
                      get_model,
                      normalize,
                      NB_CLASSES)


"""
runs model in inference mode on given input x
"""
def predict(x, model, patch_size=DEFAULT_PATCH_SIZE, nb_classes=NB_CLASSES):
    
    img_height = x.shape[0]
    img_width = x.shape[1]
    nb_channels = x.shape[2]
    
    # extend image so that it contains integer number of patches
    nb_patches_vertical = math.ceil(img_height / patch_size)
    nb_patches_horizontal = math.ceil(img_width / patch_size)
    extended_height = patch_size * nb_patches_vertical
    extended_width = patch_size * nb_patches_horizontal
    ext_x = np.zeros((extended_height, extended_width, nb_channels), dtype=np.float32)
    
    # fill extended image with mirrors
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[2 * img_width - j - 1, :, :]
    
    patches_list = []
    for i in range(nb_patches_vertical):
        for j in range(nb_patches_horizontal):
            x0, x1 = i * patch_size, (i + 1) * patch_size
            y0, y1 = j * patch_size, (j + 1) * patch_size
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    
    # model.predict() requires a numpy array
    patches = np.asarray(patches_list)
    
    # predictions (no overlap)
    patches_predict = model.predict(patches, batch_size=4)
    prediction = np.zeros((extended_height, extended_width, nb_classes), dtype=np.float32)
    for k in range(patches_predict.shape[0]):
        i = k // nb_patches_horizontal
        j = k % nb_patches_vertical
        x0, x1 = i * patch_size, (i + 1) * patch_size
        y0, y1 = j * patch_size, (j + 1) * patch_size
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    return prediction[:img_height, :img_width, :]


"""
returns an RGB image with color-coded classes based on mask

mask: mask of shape (height, width, nb_classes)
"""
def picture_from_mask(mask, threshold=0):
    colors = {
        0: [150, 150, 150],   # buildings
        1: [223, 194, 125],   # roads & tracks
        2: [27,  120, 55],    # trees
        3: [166, 219, 160],   # crops
        4: [116, 173, 209]    # water
    }
    z_order = {
        1: 3,
        2: 4,
        3: 0,
        4: 1,
        5: 2
    }
    
    pict = 255 * np.ones((3, mask.shape[1], mask.shape[2]), dtype=np.uint8)
    for i in range(1, 6):
        cl = z_order[i]
        for ch in range(3):
            pict[ch, :, :][mask[cl, :, :] > threshold] = colors[cl][ch]
    return pict


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Provide a TIF image")
        return
    
    filename = sys.argv[1]

    inp = normalize(tiff.imread(filename).transpose([1, 2, 0]))
    res = predict(inp, model, patch_size=PATCH_SIZE, nb_classes=NB_CLASSES).transpose([2, 0, 1])
    print("Processed", filename)
    res_map = picture_from_mask(res, 0.5)
    tiff.imsave('results/result_{}.tif'.format(filename[:-4]), res_map)
    print("Saved the result as", "results/result_{}.tif".format(filename[:-4]))