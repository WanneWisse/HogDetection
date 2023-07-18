from HOG import HogDetector
import numpy as np
import cv2
import os
import glob


#size of total image where the detector needs to traverse over
desired_rows = 600
desired_cols = 600
#size of detector window
detector_scale_rows = 120
detector_scale_cols = 120
#amount of steps to let the detector window travel
stride = 20
#patch size for HOG features model
model_patch_width = 64
model_patch_height = 128

#load image where the object needs to be detected
image = cv2.imread("test_img.jpg")
#resize to requested format
image = cv2.resize(image, (desired_cols, desired_rows))
#convert color to gray 3v1 dim
ref_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ref_image = np.array(ref_image)

#init HOG detector
hd = HogDetector()
#train SVM based on images of object to detect return a model in joblib
hd.train()
#when already trained load joblib model like beneath:
#hd.load_model("model.joblib")

#empty detector output folder to check results of the 20 patches with the highest prop to the class
files = glob.glob('detector_output/*')
for f in files:
    os.remove(f)

#traverse detector window over image with stride and save: (location,score) per window 
all_results = []
for rows in range(0,desired_rows-detector_scale_rows,stride):
    for cols in range(0,desired_cols-detector_scale_cols,stride):
        #get patch
        patch_to_detect = ref_image[rows:rows+detector_scale_rows, cols:cols+detector_scale_cols]
        #resize patch to be parsed for detection
        patch_to_detect = cv2.resize(patch_to_detect, (model_patch_width, model_patch_height))
        #get prop score of positive class of resized patch
        result = hd.detect(patch_to_detect)[0][1]
        #append location,score
        all_results.append(((rows,cols),result))

#get top 20 results and write patches to detector_test/ to look at performance of detector
all_results.sort(key=lambda x: x[1],reverse=True)
top20 = all_results[0:20]
for patch in top20:
    starty = patch[0][0]
    startx = patch[0][1]
    patch = ref_image[starty:starty+detector_scale_rows, startx:startx+detector_scale_cols]
    cv2.imwrite("detector_test/result" + str(starty) + str(startx) + ".jpg", patch)

#get best patch 
top1 = top20[0]
starty = top1[0][0]
startx = top1[0][1]


#bounding box around best patch on original image
cv2.rectangle(image,(startx,starty),(startx+detector_scale_cols,starty+detector_scale_rows),(0,255,0),3)
cv2.putText(image,'MeloenSterDing',(startx,starty-30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
cv2.imshow("MeloenSterDingDetector",image)
cv2.waitKey()  

#bounding box around best patch on magnitude image to check performance
Gx,Gy,Gm,Gd = hd.get_gradients(ref_image,hd.perwit_kernel_x,100)
Gm = cv2.convertScaleAbs(Gm)
Gm = cv2.cvtColor(Gm, cv2.COLOR_GRAY2BGR)
cv2.rectangle(Gm,(startx,starty),(startx+detector_scale_cols,starty+detector_scale_rows),(0,255,0),3)
cv2.putText(Gm,'MeloenSterDing',(startx,starty-30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
cv2.imshow("MeloenSterDingDetector",Gm)
cv2.waitKey()  
cv2.destroyAllWindows()