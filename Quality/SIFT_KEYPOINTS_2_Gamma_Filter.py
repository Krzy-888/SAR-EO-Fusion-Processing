import matplotlib.pyplot as plt
import numpy as np
import cv2
sift = cv2.SIFT_create()
#draw Keypoints

paths = [r"Norm/EO_UDYE_SUB_035m_gray.png",r"Norm/SAR_UDYE_SUB_GM_035m_gray.png",r"Norm/SAR_UDYE_SUB_035m_gray.png"]
name = ["Optyczne"," SAR Gamma Map 7x7", "SAR"]
fig, axes = plt.subplots(1, 3)
# fig.suptitle("SIFT")
for i,path in enumerate(paths):
    title = name[i] + f" Z"
    img = cv2.imread(path,0)
    # kp, des = sift.detectAndCompute(img[0:1600,0:1600], None)
    kp, des = sift.detectAndCompute(img[200:1000,200:1000], None)
    print(len(kp))
    points = np.array([p.pt for p in kp])
    try:
        print()
        # axes[i].scatter(points[:,0],points[:,1],s=1)
    except:
        pass
    print()
    #title = name[i] + f" Znaleziono: {len(kp)}"
    # axes[i].imshow(img[0:1000,0:1000],cmap="gray")
    axes[i].imshow(img[200:1000,200:1000],cmap="gray")
    axes[i].set_title(name[i])

plt.tight_layout()
plt.show()
