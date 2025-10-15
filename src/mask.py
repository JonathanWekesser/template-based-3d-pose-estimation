# mask.py
import cv2
import numpy as np

def masked_image(mask_pts, img):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [mask_pts], color=1)
    return img * mask
