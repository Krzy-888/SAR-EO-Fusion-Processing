import matplotlib.pyplot as plt
import numpy as np
import cv2
ORB = cv2.ORB_create(nfeatures = 800000)
#draw Keypoints

paths = [r"Norm/EO_UDYE_SUB_035m_gray.png",r"Norm/SAR_UDYE_SUB_GM_035m_gray.png",r"Norm/SAR_UDYE_SUB_035m_gray.png"]
name = ["Optyczne"," SAR Gamma Map 7x7", "SAR"]
fig, axes = plt.subplots(1, 3)
fig.suptitle("ORB")
for i,path in enumerate(paths):
    title = name[i] + f" Z"
    img = cv2.imread(path,0)
    # img_color = cv2.cvtColor(img[200:1000,200:1000],cv2.COLOR_GRAY2BGR)
    kp, des = ORB.detectAndCompute(img[200:1000,200:1000], None)
    points = np.array([p.pt for p in kp])
    title = name[i] + f" Znaleziono: {len(kp)}"
    print(len(kp))
    axes[i].scatter(points[:,0],points[:,1],s=10)
    axes[i].imshow(img[200:1000,200:1000],cmap="gray")
    axes[i].set_ylabel(name[i])

plt.tight_layout()
plt.show()
