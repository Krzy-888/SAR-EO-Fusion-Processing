import matplotlib.pyplot as plt
import numpy as np
import cv2
sift = cv2.SIFT_create()
#draw Keypoints

paths = [r"Norm/CAPELLA_C06_UIAA_10_M_Subset_gray.png",r"Norm/CAPELLA_C06_UIAA_10_M_Subset_log.png",r"Norm/CAPELLA_C06_UIAA_10_M_Subset_bad.png"]
name = ["2-98%","min-max skala dB","min-max"]
fig, axes = plt.subplots(3, 1)
fig.suptitle("SIFT")
for i,path in enumerate(paths):
    title = name[i] + f" Z"
    img = cv2.imread(path,0)
    kp, des = sift.detectAndCompute(img[1000:2000,1000:2000], None)
    print(len(kp))
    points = np.array([p.pt for p in kp])
    try:
        axes[i].scatter(points[:,0],points[:,1],s=10)
    except:
        pass
    print()
    #title = name[i] + f" Znaleziono: {len(kp)}"
    axes[i].imshow(img[500:1500,500:1500],cmap="gray")
    axes[i].set_ylabel(name[i])

plt.tight_layout()
plt.show()
