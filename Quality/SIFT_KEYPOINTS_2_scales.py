import matplotlib.pyplot as plt
import numpy as np
import cv2
sift = cv2.SIFT_create()
#draw Keypoints
#SAR
paths = [r"Norm/CAPELLA_C06_UIAA_10_M_Subset_gray.png",r"Norm/CAPELLA_C06_UIAA_1_M_Subset_gray.png",r"Norm/CAPELLA_C06_UIAA_035_M_Subset_gray.png"]
#EO
#paths = [r"Norm/IMG_PNEO4_UIAA_10m_subset_3x3_gray.png",r"Norm/IMG_PNEO4_UIAA_1m_subset_3x3_gray.png",r"Norm/IMG_PNEO4_UIAA_035m_subset_3x3_gray.png"]
parts = [100,1000,2857]
imgs = []
for i,path in enumerate(paths):
    img = cv2.imread(path,0)
    imgs.append(img[:parts[i],:parts[i]])
name = ["10m","1m","0,35m"]
fig, axes = plt.subplots(3, 1)
axes[0].set_title("SIFT SAR")
for i,im in enumerate(imgs):
    title = name[i] + f" Z"
    kp, des = sift.detectAndCompute(im, None)
    print(len(kp))
    points = np.array([p.pt for p in kp])
    
    print()
    
    axes[i].imshow(im,cmap="gray")
    try:
        axes[i].scatter(points[:,0],points[:,1],s=10)
    except:
        pass
plt.tight_layout()
plt.show()
