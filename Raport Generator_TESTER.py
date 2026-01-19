from Quality import RMSE
from Quality import Calc_and_Visual
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

datas = ["UIAA", "URWH", "UDYE"]
scales = ["10m","1m","035m","GM_035m"]
norms = ["bad","gray","log"]
methods = ["RIFT"]
grd = [10,1,0.35,0.35]

ilorazy = [grd[0]/0.35,grd[1]/0.35,grd[2]/0.35,grd[3]/0.35]

for m in methods:
    time_t = np.genfromtxt(f"report_{m}/EO_SAR_{m}_mach.csv",delimiter=',',dtype=np.float32)
    #if os.path.exists(f"report_{m}/report_{m}.html"):
    #    print("Istnieje !!!")
    #else:
    with open(f"report_{m}/report_{m}.html", "w") as raport:
        numerek = 0
        for d in datas:
            raport.write(f"<h1>Start {d}</h1>")
            for j,s in enumerate(scales):
                for n in norms:
                    T = time_t[numerek]
                    # Metody PUNKTY DOPASOWANIA
                    raport.write(f"<h3>SAR_{d}_SUB_{s}_{n}_mach</h3>")
                    try:
                        ptk_CAP = np.genfromtxt(f"report_{m}/SAR_{d}_SUB_{s}_{n}_before_mach.csv", delimiter=',',dtype=np.float32)
                        # ptk_PNEO = np.genfromtxt(f"report_{m}/EO_{d}_SUB_{s}_{n}_before_mach.csv", delimiter=',',dtype=np.float32)
                        raport.write(f"<h3>kp bf m SAR:  {len(ptk_CAP)}</h3>")                            
                        ptk_CAP = np.genfromtxt(f"report_{m}/SAR_{d}_SUB_{s}_{n}_mach.csv", delimiter=',',dtype=np.float32)
                        ptk_PNEO = np.genfromtxt(f"report_{m}/EO_{d}_SUB_{s}_{n}_mach.csv", delimiter=',',dtype=np.float32)
                    except:
                        print(f"End {d}")
                        raport.write(f"<h3 style='color:red;'>not enought points</h3>")
                        continue

                    # Punkty Referencyjnej transformacji
                    ptk_CAP_ref = np.genfromtxt(f"RefPoints/UTM_{d}_CAPELLA_ref.csv", delimiter=',',dtype=np.float32)
                    ptk_PNEO_ref = np.genfromtxt(f"RefPoints/UTM_{d}_PNEO_ref.csv", delimiter=',',dtype=np.float32)

                    # Punkty Kontrone
                    ptk_CAP_check = np.genfromtxt(f"RefPoints/UTM_{d}_CAPELLA.csv", delimiter=',',dtype=np.float32)
                    ptk_PNEO_check = np.genfromtxt(f"RefPoints/UTM_{d}_PNEO.csv", delimiter=',',dtype=np.float32)

                    # Wczzytaanie obrazu
                    if s=="GM_035m":
                        img2 = cv2.imread(f"Norm/EO_{d}_SUB_035m_gray.png",0)
                    else:
                        img2 = cv2.imread(f"Norm/EO_{d}_SUB_{s}_gray.png",0)
                    img1 = cv2.imread(f"Norm/SAR_{d}_SUB_{s}_{n}.png",0) 

                    tytuły = ["złe","dobre","wynik transformacji"]
                    ptk_PNEO_ref = ptk_PNEO_ref/ilorazy[j]
                    ptk_CAP_ref = ptk_CAP_ref/ilorazy[j]
                    ptk_PNEO_check = ptk_PNEO_check/ilorazy[j]
                    ptk_CAP_check = ptk_CAP_check/ilorazy[j]
                    # print(ptk_PNEO_ref)
                    # print(ptk_CAP_ref)
                    # Recherencyjna macierz
                    print(f"report_{m}/SAR_{d}_SUB_{s}_{n}_mach.csv")
                    print(ptk_CAP)
                    M_ref, mask = cv2.estimateAffine2D(ptk_CAP_ref, ptk_PNEO_ref)

                    # Macierz powstała w skutek dopasowania automatyczną metodą
                    M_nowa, mask = cv2.estimateAffine2D(ptk_CAP, ptk_PNEO)

                    #Właściwa maska i właściwe  CMR
                    CMR,treshold,blad,mask = RMSE.calculate_CMR_mask_new(M_ref,ptk_CAP_check,ptk_PNEO_check,ptk_CAP,ptk_PNEO)
                    #print(ptk_CAP-ptk_PNEO)
                    rmse,bled = RMSE.calculate_RMSE(M_nowa,ptk_CAP_check,ptk_PNEO_check)
                    print(rmse*grd[j])
                    print(CMR)
                    print(treshold*3*grd[j])
                    raport.write(f"<h3>RMSE[m]:  {rmse*grd[j]}, CMR[%]:  {CMR*100}, treshold:  {treshold*3*grd[j]}, T: {T}</h3>")
                    h,w = img2.shape
                    image = cv2.warpAffine(img1,M_nowa,(h,w))

                    points = [ptk_CAP[~mask],ptk_CAP[mask],ptk_PNEO[~mask],ptk_PNEO[mask]]
                    color = ['r','g']
                    fig, axes = plt.subplots(3, 1)
                    for i in range(3):
                        if i == 2:
                            axes[i].imshow(image,cmap="gray")
                        else:
                            Calc_and_Visual.show_maches_in_axis(axes[i],img1,img2,points[i],points[i+2],color[i])
                        axes[i].set_title(tytuły[i])
                                
                    plt.tight_layout()
                    plt.savefig(f"report_{m}/SAR_{d}_SUB_{s}m_{n}-EO_{d}_SUB_{s}m_gray_bad.png", dpi=300,)
                    numerek+=1
                    raport.write(f"<img src='SAR_{d}_SUB_{s}m_{n}-EO_{d}_SUB_{s}m_gray_bad.png'/>")
            print(f"End {d}")
            raport.write(f"<h1>End {d}</h1>")