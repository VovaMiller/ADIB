import numpy as np
import cv2  # OTSU thresholding


# # Expanding (backup)
# arr = ...
# dw = 2
# z = np.zeros((arr.shape[0], dw), dtype=np.float64)
# arr = np.concatenate((z, arr, z), axis=1)
# z = np.zeros((dw, arr.shape[1]), dtype=np.float64)
# arr = np.concatenate((z, arr, z), axis=0)

def get_vicinity_xy(width, height, x, y, dw):
    bx = max(0, x - dw)
    by = max(0, y - dw)
    if (x + dw) >= width:
        bx = width - (2*dw + 1)
    if (y + dw) >= height:
        by = height - (2*dw + 1)
    return bx, by

def get_mean(img_c, x, y, dw):
    bx, by = get_vicinity_xy(img_c.shape[1], img_c.shape[0], x, y, dw)
    return img_c[by:by+2*dw+1, bx:bx+2*dw+1].mean()

def get_deviation(img_c, x, y, dw):
    bx, by = get_vicinity_xy(img_c.shape[1], img_c.shape[0], x, y, dw)
    return img_c[by:by+2*dw+1, bx:bx+2*dw+1].std()

def get_min(img_c, x, y, dw):
    bx, by = get_vicinity_xy(img_c.shape[1], img_c.shape[0], x, y, dw)
    return img_c[by:by+2*dw+1, bx:bx+2*dw+1].min()

def get_max(img_c, x, y, dw):
    bx, by = get_vicinity_xy(img_c.shape[1], img_c.shape[0], x, y, dw)
    return img_c[by:by+2*dw+1, bx:bx+2*dw+1].max()

def integrate_matrix(img):
    for y in range(1, img.shape[0]):
        img[y, 0] += img[y-1, 0]
    for x in range(1, img.shape[1]):
        img[0, x] += img[0, x-1]
    for x in range(1, img.shape[1]):
        for y in range(1, img.shape[0]):
            img[y, x] += (img[y-1, x] + img[y, x-1]) - img[y-1, x-1]

# TODO: DIY
def get_threshold_otsu(img, x, y, dw):
    bx, by = get_vicinity_xy(img.shape[1], img.shape[0], x, y, dw)
    t, _ = cv2.threshold(img[by:by+2*dw+1, bx:bx+2*dw+1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t

def adbin_mean(img):
    W = 15    # block size
    C = 11    # parameter
    
    assert (W % 2) == 1
    assert (img.shape[0] >= W) and (img.shape[1] >= W)
    dw = W // 2
    
    img_r = np.array(img, copy=True)                     # Result holder.
    img_c = np.array(img, dtype=np.float64, copy=True)   # Copy with proper type.
    
    # Iterating over each pixel.
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            mean = get_mean(img_c, x, y, dw)
            img_r[y, x] = 255 if (img[y, x] >= mean - C) else 0

    return img_r

def adbin_Niblack(img):
    W = 15    # block size
    C = 10    # parameter
    k = -0.2  # constant
    
    assert (W % 2) == 1
    assert (img.shape[0] >= W) and (img.shape[1] >= W)
    dw = W // 2
    
    img_r = np.array(img, copy=True)                     # Result holder.
    img_c = np.array(img, dtype=np.float64, copy=True)   # Copy with proper type.
    
    # Iterating over each pixel.
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            mean = get_mean(img_c, x, y, dw)
            std = get_deviation(img_c, x, y, dw)
            img_r[y, x] = 255 if (img[y, x] >= mean + k*std - C) else 0

    return img_r

def adbin_Sauvola(img):
    W = 15    # block size
    k = 0.35  # parameter [0.2, 0.5]
    R = 128   # constant
    
    assert (W % 2) == 1
    assert (img.shape[0] >= W) and (img.shape[1] >= W)
    dw = W // 2
    
    img_r = np.array(img, copy=True)                     # Result holder.
    img_c = np.array(img, dtype=np.float64, copy=True)   # Copy with proper type.
    
    # Iterating over each pixel.
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            mean = get_mean(img_c, x, y, dw)
            std = get_deviation(img_c, x, y, dw)
            img_r[y, x] = 255 if (img[y, x] >= mean * (1 + k * ((std / R) - 1))) else 0

    return img_r

def adbin_TRSingh(img):
    W = 15    # block size
    k = 0.35  # parameter [0, 1]

    assert (W % 2) == 1
    assert (img.shape[0] >= W) and (img.shape[1] >= W)
    dw = W // 2
    
    img_r = np.array(img, copy=True)                     # Result holder.
    img_c = np.array(img, dtype=np.float64, copy=True)   # Copy with proper type.
    
    # Iterating over each pixel.
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            mean = get_mean(img_c, x, y, dw)
            d = mean - img[y, x]
            img_r[y, x] = 255 if (img[y, x] >= mean * (1 + k * (d/(255-d) - 1))) else 0

    return img_r

def adbin_LAAB(img):
    W = 15    # block size
    k = 0.55  # parameter (0.5, 0.6)

    assert (W % 2) == 1
    assert (img.shape[0] >= W) and (img.shape[1] >= W)
    dw = W // 2
    
    img_r = np.array(img, copy=True)                      # Result holder.
    img_int = np.array(img, dtype=np.float64, copy=True)  # Copy with proper type.
    img_int *= -1
    img_int += 255
    img_int /= 255
    integrate_matrix(img_int)
    
    # Iterating over each pixel.
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # Getting original pixel value.
            if (x == 0) and (y == 0):
                g = img_int[0, 0]
            elif (x == 0):
                g = img_int[y, 0] - img_int[y-1, 0]
            elif (y == 0):
                g = img_int[0, x] - img_int[0, x-1]
            else:
                g = img_int[y, x] - (img_int[y-1, x] + img_int[y, x-1]) + img_int[y-1, x-1]
            
            # Getting mean of original image.
            bx, by = get_vicinity_xy(img.shape[1], img.shape[0], x, y, dw)
            if (bx == 0) and (by == 0):
                mean = img_int[W-1, W-1]
            elif (bx == 0):
                mean = img_int[by+W-1, W-1] - img_int[by-1, W-1]
            elif (by == 0):
                mean = img_int[W-1, bx+W-1] - img_int[W-1, bx-1]
            else:
                mean = (img_int[by+W-1, bx+W-1]
                        - (img_int[by-1, bx+W-1] + img_int[by+W-1, bx-1])
                        + img_int[by-1, bx-1])
            mean /= W*W
            
            # Obtaining result pixel.
            d = (g - mean) / (1 - mean)
            v = k*(1+d)/(1-d)
            img_r[y, x] = 255 if (1 - 2*v) > 0 else 0

    return img_r

def adbin_Bernsen(img):
    W = 31    # block size
    L = 15    # parameter

    assert (W % 2) == 1
    assert (img.shape[0] >= W) and (img.shape[1] >= W)
    dw = W // 2
    
    img_r = np.array(img, copy=True)                     # Result holder.
    img_c = np.array(img, dtype=np.float64, copy=True)   # Copy with proper type.
    
    # Iterating over each pixel.
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            Imin = get_min(img_c, x, y, dw)
            Imax = get_max(img_c, x, y, dw)
            if (Imax - Imin) >= L:
                T = (Imax + Imin) / 2
            else:
                T = get_threshold_otsu(img, x, y, dw)
            img_r[y, x] = 255 if img[y, x] >= T else 0

    return img_r

def adbin_OISingh(img):
    W = 15    # block size
    k = 0.7   # parameter (0, 1)

    assert (W % 2) == 1
    assert (img.shape[0] >= W) and (img.shape[1] >= W)
    dw = W // 2
    
    img_r = np.array(img, copy=True)                     # Result holder.
    img_c = np.array(img, dtype=np.float64, copy=True)   # Copy with proper type.
    
    # Iterating over each pixel.
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            Imin = get_min(img_c, x, y, dw)
            Imax = get_max(img_c, x, y, dw)
            mean = get_mean(img_c, x, y, dw)
            img_r[y, x] = 255 if img[y, x] >= k*(mean + (Imax - Imin) * img_c[y, x] / 255) else 0

    return img_r

