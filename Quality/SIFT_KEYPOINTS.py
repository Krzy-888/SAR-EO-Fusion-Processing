import matplotlib.pyplot as plt
import numpy as np
import cv2
sift = cv2.SIFT_create()
#draw Keypoints
def draw_cross_keypoints(img, keypoints, color):
    """ Draw keypoints as crosses, and return the new image with the crosses. """
    img_kp = img.copy()  # Create a copy of img

    # Iterate over all keypoints and draw a cross on evey point.
    for kp in keypoints:
        x, y = kp.pt  # Each keypoint as an x, y tuple  https://stackoverflow.com/questions/35884409/how-to-extract-x-y-coordinates-from-opencv-cv2-keypoint-object

        x = int(round(x))  # Round an cast to int
        y = int(round(y))

        # Draw a cross with (x, y) center
        cv2.drawMarker(img_kp, (x, y), color, markerType=cv2.MARKER_DIAMOND, markerSize=5, thickness=1, line_type=cv2.LINE_8)

    return img_kp  # Return the image with the drawn crosses.
paths = [r"Norm/CAPELLA_C05_URRC_Subset1000_2_gray.png",r"Norm/CAPELLA_C05_URRC_Subset1000_2_log.png",r"Norm/CAPELLA_C05_URRC_Subset1000_2_bad.png"]
name = ["normalizacja 2-98%","min-max skala dB","normalizacja min-max"]
fig, axes = plt.subplots(3, 1)
for i,path in enumerate(paths):
    title = name[i] + f" Z"
    img = cv2.imread(path,0)
    img_color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    kp, des = sift.detectAndCompute(img, None)
    img_2 = draw_cross_keypoints(img_color, kp, (0,255,0))
    title = name[i] + f" Znaleziono: {len(kp)}"
    axes[i].imshow(img_2)
    axes[i].set_title(title)

plt.tight_layout()
plt.show()
