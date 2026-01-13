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
        
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * factor),
                int(dataset.width * factor)
            ),
            resampling=Resampling.bilinear
        )
        zasieg =[4971,7828,5462,8319]
        pp1 = int(zasieg[0]*factor)
        pk1 = int(zasieg[1]*factor)
        pp2 = int(zasieg[2]*factor)
        pk2 = int(zasieg[3]*factor)

    return data[pp2:pk2,pp1:pk1]

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
        if len(array) == 3:
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
paths = [r"CAPELLA_C09_URWH_SUB.tif",r"IMG_PNEO4_URWH_SUB.tif"]
names = [10,1,"035"]
factors = [0.35/10,0.35/1,1]
#   resample AND PROCESS

for i,path in enumerate(paths):
    name = path.split("\\")[-1].replace(".tif","")
    with rasterio.open(path) as src:
        for j,f in enumerate(factors):
            img = resample(path,f)

            if len(img) == 3:
                norm = normalize_to_uint8(img)
            else:
                norm,log,bnorm = normalize_to_uint8(img)
            name2 = f"{name}_{names[j]}m"
            print(name2)
            print("\t***Before***")
            print(img.shape)
            print(img.max())
            print(img.min())
            print("\t***After***")
            print(norm.shape)
            print(norm.max())
            print(norm.min())
            if len(img) == 1:
                print("\t***Log***\t")
                print(log.shape)
                print(log.max())
                print(log.min())
                print("\t***bad***\t")
                print(bnorm.shape)
                print(bnorm.max())
                print(bnorm.min())
            if len(img) == 3:
                blue = norm[2]
                green = norm[1]
                red = norm[0]
                bgr = np.stack([blue, green, red], axis=-1)
                norm_2 = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
                out_g = f"Norm\\{name2}_gray_n.png"
                cv2.imwrite(out_g,norm_2)
            else:
                gray = norm.mean(axis=0).astype(np.uint8)
                gray_l = log.mean(axis=0).astype(np.uint8)
                gray_b = bnorm.mean(axis=0).astype(np.uint8)
                out_g = f"Norm\\{name2}_gray_n.png"
                out_l = f"Norm\\{name2}_log_n.png"
                out_b = f"Norm\\{name2}_bad_n.png"
                cv2.imwrite(out_g,gray)
                cv2.imwrite(out_l,gray_l)
                cv2.imwrite(out_b,gray_b)
# SAVE
print("DONE")
