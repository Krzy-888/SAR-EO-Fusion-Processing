import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2

def calculate_RMSE(Macierz,pkt_zrod,pkt_ref):
    """
    Docstring for calculate_RMSE
    
    :param Macierz: Description
    :param pkt_zrod: Description
    :param pkt_ref: Description
    """
    # Dodanie Skali
    pkt_po_trans = np.c_[pkt_zrod,np.ones(pkt_zrod.shape[0])]
    # Transpozycja Macierzy
    M = Macierz.T
    # Macierzowe mnozenie  
    pkt_po_trans = np.dot(pkt_po_trans, M)
    # Roznica
    roznica = pkt_ref - pkt_po_trans
    # Odleglosc
    odleglosc = np.sum(roznica**2, axis=1)
    # RMSE axis=1 zapewnia 
    RMSE = np.sqrt(sum(odleglosc)/(len(odleglosc)-1))
    return RMSE, odleglosc

def calculate_CMR(pkt_ref1,pkt_ref2,src_pts,dst_pts,treshold):
    """
    Docstring for calculate_CMR
    
    :param pkt_ref1: Description
    :param pkt_ref2: Description
    :param src_pts: Description
    :param dst_pts: Description
    :param treshold: Description
    """
    
    M,_ = cv2.estimateAffine2D(pkt_ref1,pkt_ref2)
    rmse,blad = calculate_RMSE(M,src_pts,dst_pts)
    corr = blad[blad <= treshold] 
    CMR = len(corr)/len(blad)*100
    return CMR,rmse,blad

def calculate_CMR_mask(pkt_ref1,pkt_ref2,src_pts,dst_pts):
    """
    Docstring for calculate_CMR
    
    :param pkt_ref1: Description
    :param pkt_ref2: Description
    :param src_pts: Description
    :param dst_pts: Description
    :param treshold: Description
    """
    
    M,_ = cv2.estimateAffine2D(pkt_ref1,pkt_ref2)
    rmse,blad = calculate_RMSE(M,pkt_ref1,pkt_ref2)
    treshold = max(blad)
    rmse,blad = calculate_RMSE(M,src_pts,dst_pts)
    corr = blad[blad <= treshold] 
    mask = blad <= treshold
    CMR = len(corr)/len(blad)*100
    return CMR,rmse,blad,mask



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