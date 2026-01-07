import matplotlib.pyplot as plt
import numpy as np
import cv2
sift = cv2.SIFT_create()
#draw Keypoints
#SAR
#paths = [r"Norm/SAR_UIAA_SUB_10m_gray.png",r"Norm/SAR_UIAA_SUB_1m_gray.png",r"Norm/SAR_UIAA_SUB_035m_gray.png"]
#EO
paths = [r"Norm/EO_UIAA_SUB_10m_gray.png",r"Norm/EO_UIAA_SUB_1m_gray.png",r"Norm/EO_UIAA_SUB_035m_gray.png"]
parts = [100,1000,2857]
imgs = []
for i,path in enumerate(paths):
    img = cv2.imread(path,0)
    imgs.append(img[:parts[i],:parts[i]])
name = ["10m","1m","0,35m"]
fig, axes = plt.subplots(3, 1)
axes[0].set_title("SIFT")
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
