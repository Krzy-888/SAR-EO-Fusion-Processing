import RMSE_IMG
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

paths = [r"Norm/SAR_UDYE_SUB_GM_035m_log.png",r"Norm/SAR_UDYE_SUB_035m_log.png",r"Norm/SAR_UDYE_SUB_GM_035m_bad.png",r"Norm/SAR_UDYE_SUB_035m_bad.png",r"Norm/SAR_UDYE_SUB_GM_035m_gray.png",r"Norm/SAR_UDYE_SUB_035m_gray.png"]
name = ["SAR Gamma Map 7x7 log", "SAR", "SAR Gamma Map 7x7 bad", "SAR", "SAR Gamma Map 7x7 gray", "SAR"]

img_pleiades = cv2.imread(r"Norm/EO_UDYE_SUB_035m_gray.png",0)
img_pleiades = img_pleiades[200:1000,200:1000]

for i,path in enumerate(paths):
    img = cv2.imread(path,0)
    img = img[200:1000,200:1000]
    wynik = RMSE_IMG.calculte_Image_RMSE(img_pleiades,img)
    print(f"RMSE {name[i]}:\t{wynik}")
    mae = np.mean(np.abs(img_pleiades.astype(np.float32) - img.astype(np.float32)))
    print(f"MAE {name[i]}:\t{mae}")
    mse = np.mean((img_pleiades.astype(np.float32) - img.astype(np.float32))**2)
    psnr = 10 * np.log10(255**2 / mse)
    print(f"MSE {name[i]}:\t{mse}")
    print(f"PSNR {name[i]}:\t{psnr}")
    val, ssim_map = ssim(img_pleiades, img, full=True)
    print(f"SSIM {name[i]}:\t{val}")