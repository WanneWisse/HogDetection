import numpy as np
import cv2
import os
import glob
from HOG import HogDetector

#remove all files from folders
files = glob.glob('detector_test/*')
for f in files:
    os.remove(f)
files = glob.glob("train_preprocessed/not/*")
for f in files:
    os.remove(f)
files = glob.glob("train_preprocessed/is/*")
for f in files:
    os.remove(f)

#patch size for HOG features model
model_patch_width = 64
model_patch_height = 128

#for files from negative class (can be background), first resize and crop then iterate with stride and get small patches to generate more data to train
files = glob.glob('train/not/*')
patch_size = 200
stride = 100
crop=800
for f in files:
    image = cv2.imread(f.title())
    image = cv2.resize(image, (crop, crop))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.array(image)
    #save also the fullsize image as train data, resize to model_patch_size
    cv2.imwrite("train_preprocessed/not/result" + os.path.basename(f), cv2.resize(image, (model_patch_width, model_patch_height)))
    #save small patches from fullsize image as train data
    for rows in range(0,crop-patch_size,stride):
        for cols in range(0,crop-patch_size,stride):
            patch = image[rows:rows+patch_size, cols:cols+patch_size]
            patch= cv2.resize(patch, (model_patch_width, model_patch_height))
            cv2.imwrite("train_preprocessed/not/result" + str(rows) + str(cols) + os.path.basename(f), patch)

#for files from positive class (make sure zoomed in at the object), same process but because already zoomed in detectorscale much larger.
#to still generate enough train data, small stride
files = glob.glob('train/is/*')
detector_scale = 300
stride = 20
crop = 400
for f in files:
    image = cv2.imread(f.title())
    image = cv2.resize(image, (crop, crop))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.array(image)
    cv2.imwrite("train_preprocessed/is/result" + os.path.basename(f), cv2.resize(image, (model_patch_width, model_patch_height)))
    for rows in range(0,crop-detector_scale,stride):
        for cols in range(0,crop-detector_scale,stride):
            patch = image[rows:rows+detector_scale, cols:cols+detector_scale]
            patch= cv2.resize(patch, (model_patch_width, model_patch_height))
            cv2.imwrite("train_preprocessed/is/result" + str(rows) + str(cols) + os.path.basename(f), patch)

 