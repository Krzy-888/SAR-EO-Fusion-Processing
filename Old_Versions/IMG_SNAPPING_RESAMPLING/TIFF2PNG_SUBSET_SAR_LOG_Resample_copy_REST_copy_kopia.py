#   LIBRARIES
import rasterio
from rasterio.enums import Resampling
import numpy as np
import cv2

# Resampling https://rasterio.readthedocs.io/en/stable/topics/resampling.html
def resample(path, factor):
    '''
        Docstring for resample

        :param dataset: RasterioIMG
        :param factor: Factor
    '''
    with rasterio.open(path) as dataset:
        # resample data to target shape
        data = dataset.read()
    return data

#   NORMALIZE
def normalize_to_uint8(array):
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = array.astype(np.float32)
    
    # Logarytmic scale Db
    if len(array) ==1:
            log_array = 20*np.log10(array)
            min_val = log_array.min()
            max_val = log_array.max()
            log_array = (log_array - min_val) / (max_val - min_val)
            log_array *= 255 
    
    
    for i in range(len(array)):
        #clasic Normalization for EO
        if len(img) == 3:
            min_val = array[i].min()
            max_val = array[i].max()
        else:
            #that pice of shit that may be helpful For SAR
            min_val = array[i].min()
            max_val = array[i].max()
            bad_norm = (array - min_val) / (max_val - min_val)
            bad_norm *= 255
            min_val, max_val = np.percentile(array, (2, 98))
            array = np.clip(array, min_val, max_val)
        array[i] = (array[i] - min_val) / (max_val - min_val)      

    array *= 255
    if len(array) == 1:
        return array.astype(np.uint8), log_array.astype(np.uint8),bad_norm.astype(np.uint8)
    else:
         return array.astype(np.uint8)

#   DATASET
paths = [r"Speckle/SAR_UDYE_SUB_GM.tif",r"Speckle/SAR_URRC_SUB_GM.tif"]
names = [10,1,"035"]
#factors = [0.35/10,0.35/1,1]

#   resample AND PROCESS

for i,path in enumerate(paths):
    name = path.split("\\")[-1].replace(".tif","")
    with rasterio.open(path) as src:
        img = src.read()
       # for j,f in enumerate(factors):
        norm,log,bnorm = normalize_to_uint8(img)
        name2 = f"{name}_035m"
        print(name2)
        print("\t***Before***")
        print(img.shape)
        print(img.max())
        print(img.min())
        print("\t***After***")
        print(norm.shape)
        print(norm.max())
        print(norm.min())
        print("\t***Log***\t")
        print(log.shape)
        print(log.max())
        print(log.min())
        print("\t***bad***\t")
        print(bnorm.shape)
        print(bnorm.max())
        print(bnorm.min())
        gray = norm.mean(axis=0).astype(np.uint8)
        gray_l = log.mean(axis=0).astype(np.uint8)
        gray_b = bnorm.mean(axis=0).astype(np.uint8)
        # gray = gray[pp2:pk2,pp1:pk1]
        # gray_l = gray_l[pp2:pk2,pp1:pk1]
        # gray_b = gray_b[pp2:pk2,pp1:pk1]
        out_g = f"{name2}_gray.png"
        out_l = f"{name2}_log.png"
        out_b = f"{name2}_bad.png"
        print(out_g,out_l,out_b)
        cv2.imwrite(out_g,gray)
        cv2.imwrite(out_l,gray_l)
        cv2.imwrite(out_b,gray_b)
        print("git")
# SAVE
print("DONE")
