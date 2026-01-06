import matplotlib.pyplot as plt
import numpy as np
import cv2

paths = [r"Norm/CAPELLA_C05_URRC_Subset1000_2_gray.png",r"Norm/CAPELLA_C05_URRC_Subset1000_2_log.png",r"Norm/CAPELLA_C05_URRC_Subset1000_2_bad.png"]
name = ["2-98%","min-max skala dB","min-max"]

fig, axes = plt.subplots(2, 3)
for i,path in enumerate(paths):
    
    title = name[i]
    img = cv2.imread(path,0)
    axes[0,i].imshow(img,cmap="gray")
    axes[0,i].set_title(title)

    axes[1,i].hist(img.ravel(), bins=256)

plt.tight_layout()
plt.show()

        

