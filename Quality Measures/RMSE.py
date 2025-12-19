import cv2
import numpy as np

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
    RMSE = np.sqrt(np.mean(odleglosc))
    return RMSE, odleglosc