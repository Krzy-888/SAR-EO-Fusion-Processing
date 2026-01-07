import matplotlib.pyplot as plt
import numpy as np
import cv2
ORB = cv2.ORB_create(10000000)
#draw Keypoints

paths = [r"Norm/SAR_URRC_SUB_035m_gray.png",r"Norm/SAR_URRC_SUB_035m_log.png",r"Norm/SAR_URRC_SUB_035m_bad.png"]
name = ["2-98%","min-max skala dB","min-max"]
fig, axes = plt.subplots(3, 1)
fig.suptitle("ORB")
for i,path in enumerate(paths):
    title = name[i] + f" Z"
    img = cv2.imread(path,0)
    img = img[1000:2000,1000:2000]
    img_color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    kp, des = ORB.detectAndCompute(img, None)
    print(len(kp))
    points = np.array([p.pt for p in kp])
    axes[i].scatter(points[:,0],points[:,1],s=10)
    print()
    #title = name[i] + f" Znaleziono: {len(kp)}"
    axes[i].imshow(img,cmap="gray")
    axes[i].set_ylabel(name[i])

plt.tight_layout()
plt.show()
