#   LIBRARIES
import rasterio
import numpy as np
import cv2

#   NORMALIZE
def normalize_to_uint8(array):
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = array.astype(np.float32)
    for i in range(len(array)):
        min_val = array[i].min()
        max_val = array[i].max()
        array[i] = (array[i] - min_val) / (max_val - min_val)
    array *= 255
    return array.astype(np.uint8)

#   SUBSET AND PROCESS
def subset_and_save(path):
    with rasterio.open(path[0]) as src:
        img = src.read()
        subset = img[0:len(img),path[2][0]:path[2][1], path[1][0]:path[1][1]]
        norm = normalize_to_uint8(subset)
        name = path[0].split("\\")[-1].replace(".tif","")
        print(name)
        print("\t***Before***")
        print(img.shape)
        print(img.max())
        print(img.min())
        print("\t***After***")
        print(norm.shape)
        print(norm.max())
        print(norm.min())
        #cv2.imwrite(path[0].split("\\")[-1].replace(".tif","_Norm.png"),norm)
        # norm.shape: (2, H, W)
        if path[3]=="RGB+Gray":
            b = norm[2]
            g = norm[1]
            r = norm[0]
            bgr = np.stack([b, g, r], axis=-1)
            norm_2 = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            out_g = "Norm\\".join(name).join("_gray.png")
            out_rgb ="Norm\\".join(name).join("_rgb.png")
            cv2.imwrite(out,norm_2)
            cv2.imwrite(out,bgr)
        if path[3]=="RGB2Gray":
            blue = norm[2]
            green = norm[1]
            red = norm[0]
            bgr = np.stack([blue, green, red], axis=-1)
            norm_2 = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            out_g = "Norm\\".join(name).join("_gray.png")
            cv2.imwrite(out_g,norm_2)
        if path[3]=="Gray":
            out_g = "Norm\\".join(name).join("_gray.png")
            cv2.imwrite(out_g,norm)
        
#   DATASET
path = [[r"TEST_SENTINEL_2",[100,612],[100,612],"RGB+Gray"],
        [r"TEST_SENTINEL_1",[100,612],[100,612],"Gray"]]
subset_and_save(path)
print("DONE")