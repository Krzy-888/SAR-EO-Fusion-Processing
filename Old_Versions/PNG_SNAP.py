#   LIBRARIES
import rasterio
from rasterio.enums import Resampling
import numpy as np
import cv2


#   DATASET
DATA = "UIAA"
scales = ["10","1","035"]
norms = ["gray","log","bad"]
ranges = [[[38,138],[82,182]],[[375,1375],[815,1815]],[[1071,3928],[2328,5185]]]

for s,scale in enumerate(scales):
    print(f"New_Name/EO_{DATA}_SUB_{scale}m_gray.png")
    eo_img = cv2.imread(f"New_Name/EO_{DATA}_SUB_{scale}m_gray.png",0)
    print(eo_img.shape)
    new_eo = eo_img[ranges[s][1][0]:ranges[s][1][1],ranges[s][0][0]:ranges[s][0][1]]
    print(new_eo.shape)
    out_eo = f"SNAPPING/EO_{DATA}_SUB_{scale}m_gray.png"
    cv2.imwrite(out_eo,new_eo)
    for norm in norms:
        sar_img = cv2.imread(f"New_Name/SAR_{DATA}_SUB_{scale}m_{norm}.png",0)
        print(sar_img.shape)
        new_sar = sar_img[ranges[s][1][0]:ranges[s][1][1],ranges[s][0][0]:ranges[s][0][1]]
        print(new_sar.shape)
        out_sar = f"SNAPPING/SAR_{DATA}_SUB_{scale}m_{norm}.png"
        cv2.imwrite(out_sar,new_sar)

print("DONE")
