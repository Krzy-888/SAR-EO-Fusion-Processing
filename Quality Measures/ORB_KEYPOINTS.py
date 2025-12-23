import matplotlib.pyplot as plt
import numpy as np
import cv2
ORB = cv2.ORB_create(nfeatures = 800000)
#draw Keypoints

paths = [r"Norm/CAPELLA_C05_URRC_Subset1000_2_gray.png",r"Norm/CAPELLA_C05_URRC_Subset1000_2_log.png",r"Norm/CAPELLA_C05_URRC_Subset1000_2_bad.png"]
name = ["normalizacja 2-98%","min-max skala dB","normalizacja min-max"]
fig, axes = plt.subplots(3, 1)
for i,path in enumerate(paths):
    title = name[i] + f" Z"
    img = cv2.imread(path,0)
    img_color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    kp, des = ORB.detectAndCompute(img, None)
    points = np.array([p.pt for p in kp])
    title = name[i] + f" Znaleziono: {len(kp)}"
    print(len(kp))
    axes[i].scatter(points[:,0],points[:,1],s=10)
    axes[i].imshow(img,cmap="gray")
    axes[i].set_title(title)

plt.tight_layout()
plt.show()
