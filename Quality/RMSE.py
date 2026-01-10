import cv2
import numpy as np

def calculate_RMSE_Homo(H, pkt_zrod, pkt_ref):
    """
    RMSE dla transformacji homograficznej

    :param H: macierz homografii 3x3
    :param pkt_zrod: punkty źródłowe (N, 2)
    :param pkt_ref: punkty referencyjne (N, 2)
    """

    # 1. Współrzędne jednorodne
    pkt_h = np.c_[pkt_zrod, np.ones(pkt_zrod.shape[0])]

    # 2. Transformacja homograficzna
    pkt_trans = (H @ pkt_h.T).T   # (N, 3)

    # 3. Normalizacja (x/w, y/w)
    pkt_trans_xy = pkt_trans[:, :2] / pkt_trans[:, 2][:, np.newaxis]

    # 4. Różnice
    roznica = pkt_ref - pkt_trans_xy

    # 5. Odległość euklidesowa^2
    odleglosc2 = np.sum(roznica**2, axis=1)

    # 6. RMSE
    RMSE = np.sqrt(np.mean(odleglosc2))

    return RMSE, odleglosc2



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
