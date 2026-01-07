from Quality import RMSE_IMG
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
#EO Data
EO_Paths = [r"Norm/IMG_PNEO4_UIAA_SUB_10m_gray.png",r"Norm/IMG_PNEO4_UIAA_SUB_1m_gray.png",r"Norm/IMG_PNEO4_UIAA_SUB_035m_gray.png"]

#SAR Data
SAR_paths = [r"Norm/CAPELLA_C06_UIAA_SUB_10m_bad.png",r"Norm/CAPELLA_C06_UIAA_SUB_1m_bad.png",r"Norm/CAPELLA_C06_UIAA_SUB_035m_bad.png"]
name = ["10m","1m","0,35m"]
size = [100,1000,2857]
for i,path in enumerate(SAR_paths):
    img_sar = cv2.imread(path,0)
    img_eo = cv2.imread(EO_Paths[i],0)
    img_sar = img_sar[0:size[i],0:size[i]]
    img_eo = img_eo[0:size[i],0:size[i]]
    wynik = RMSE_IMG.calculte_Image_RMSE(img_eo,img_sar)
    print(f"RMSE {name[i]}:\t{wynik}")
    #mae = np.mean(np.abs(img_pleiades.astype(np.float32) - img.astype(np.float32)))
    #print(f"MAE {name[i]}:\t{mae}")
    #mse = np.mean((img_pleiades.astype(np.float32) - img.astype(np.float32))**2)
    #psnr = 10 * np.log10(255**2 / mse)
    #print(f"MSE {name[i]}:\t{mse}")
    #print(f"PSNR {name[i]}:\t{psnr}")
    val, ssim_map = ssim(img_eo, img_sar, full=True)
    print(f"SSIM {name[i]}:\t{val}")