import matplotlib.pyplot as plt
import numpy as np
import cv2
sift = cv2.SIFT_create()
#draw Keypoints

paths = [r"Norm/CAPELLA_C05_URRC_Subset1000_2_gray.png",r"Norm/CAPELLA_C05_URRC_Subset1000_2_log.png",r"Norm/CAPELLA_C05_URRC_Subset1000_2_bad.png"]
name = ["2-98%","min-max skala dB","min-max"]
fig, axes = plt.subplots(3, 1)
fig.suptitle("SIFT")
for i,path in enumerate(paths):
    title = name[i] + f" Z"
    img = cv2.imread(path,0)
    img_color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    kp, des = sift.detectAndCompute(img, None)
    print(len(kp))
    points = np.array([p.pt for p in kp])
    axes[i].scatter(points[:,0],points[:,1],s=10)
    print()
    #title = name[i] + f" Znaleziono: {len(kp)}"
    axes[i].imshow(img,cmap="gray")
    axes[i].set_ylabel(name[i])

plt.tight_layout()
plt.show()
