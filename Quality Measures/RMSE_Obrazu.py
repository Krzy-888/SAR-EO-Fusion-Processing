import cv2
import numpy as np

def calculte_Image_RMSE(img_1,img_2):
    diff = img_1-img_2
    sqr_diff = diff*diff
    mean = sqr_diff.mean()
    RMSE = np.sqrt(mean)
    return RMSE

