import matplotlib.pyplot as plt
import numpy as np
import cv2


def mach_imgpoints(img1,img2,points2):
    col_num = img1.shape[1] + img2.shape[1]
    row_num = max(img1.shape[0], img2.shape[0])
    new_img = np.zeros((row_num, col_num), dtype=np.uint8)
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:] = img2
    points2_x = points2 + [img1.shape[1],0]
    new_points = points2_x
    return new_img,new_points

def show_maches_in_axis(axis,img1,img2,points1,points2,colors):
    img,points_new = mach_imgpoints(img1,img2,points2)
    axis.imshow(img,cmap="gray")
    for i in range(len(ptk_PNEO)):
        if len(colors)>3:
            color = colors[i]
        else:
            color = colors
        axis.plot([points1[i,0],points_new[i,0]],[points1[i,1],points_new[i,1]],c=color)
        axis.scatter(points1[i,0],points1[i,1],c=color)
        axis.scatter(points_new[i,0],points_new[i,1],c=color)