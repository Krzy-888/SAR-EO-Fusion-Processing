import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.collections import LineCollection

def show_maches_in_axis_2(axis, img1, img2, points1, points2, color='r'):
    img, points_new = mach_imgpoints(img1, img2, points2)
    axis.imshow(img, cmap="gray")

    # Linie: (N, 2, 2)
    lines = np.stack(
        [
            np.stack([points1[:, 0], points1[:, 1]], axis=1),
            np.stack([points_new[:, 0], points_new[:, 1]], axis=1)
        ],
        axis=1
    )

    lc = LineCollection(lines, colors=color, linewidths=1)
    axis.add_collection(lc)

    # Punkty – tylko DWA scattery
    axis.scatter(points1[:, 0], points1[:, 1], c=color, s=5)
    axis.scatter(points_new[:, 0], points_new[:, 1], c=color, s=5)








def mach_imgpoints(img1,img2,points2):
    """
    Docstring for mach_imgpoints
    
    :param img1: Description
    :param img2: Description
    :param points2: Description
    """
    col_num = img1.shape[1] + img2.shape[1]
    row_num = max(img1.shape[0], img2.shape[0])
    new_img = np.zeros((row_num, col_num), dtype=np.uint8)
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:] = img2
    points2_x = points2 + [img1.shape[1],0]
    new_points = points2_x
    return new_img,new_points

def show_maches_in_axis(axis,img1,img2,points1,points2,colors):
    """
    Docstring for show_maches_in_axis

    :param axis: Description
    :param img1: Description
    :param img2: Description
    :param points1: Description
    :param points2: Description
    :param colors: Description
    """
    img,points_new = mach_imgpoints(img1,img2,points2)
    axis.imshow(img,cmap="gray")
    for i in range(len(points2)):
        if len(colors)>3:
            color = colors[i]
        else:
            color = colors
        axis.plot([points1[i,0],points_new[i,0]],[points1[i,1],points_new[i,1]],c=color)
        axis.scatter(points1[i,0],points1[i,1],c=color)
        axis.scatter(points_new[i,0],points_new[i,1],c=color)

def resamplepoints(data,default_grd,scales):
    """
    Docstring for przeskaluj_pkt

    :param data: Description
    :param default_grd: Description
    :param scales: Description
    """
    data_new = []
    for scale in scales:
        denominator = scale/default_grd
        data_new.append(data/denominator)
    return data_new

def show_points(ax,points):
    ax.scatter(points[0],points[1])