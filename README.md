# HogDetection
This was a hobby project performed after following a computer vision course during my masters.
During this course we learned about different classifiers like: Color models, SIFT anddd HOG which stands for histogram of oriented gradients! 
I really liked the idea of taking gradients of pixels and got interested in trying this out myself. The histogram of oriented gradients is literaly what the name says, only then with some normalisation :P.
The advantages of this algorithm are that it does NOT depend on color; only on the gradient of the pixels and the feature space is small.

## Steps
The algorithm consists of the following steps:
1. Take the gradient in the x and y direction per pixel. This can be done by looking at neighboring pixels. In HOG they used Sobel and Perwit matrices to do this.
In the implementation I used Perwit but you can easily change this by changing the numbers, the size stays the same!
2. Next you can calculate the magintude and direction of the x and y gradients per pixel. For the maginitude I used the Euclidian between the x and y gradients. For the direction I used the atan2 between the x and y gradients.
3. In this step you group pixels together in cells of a particular size. The idea of HOG is that an image is defined by gradients of different parts. Using this way we can reduce the size of the feature vector while keeping the important information
4. Per cell we create a histogram. A histogram consists of bins which represent the direction of the gradients. We divide the range of arctan2 which is 0-360 in 9 equal parts. When having a pixel, we look to which bin his gradient direction is the closest and add the magnitude to the bin, higher maginitude means more importance.
5. Next we stride over groups of cells and call these groups blocks. For every block we normalise using the L2 norm. Having the histograms per cell we have to deal with problems like contrast and light. These can cause very different values for the magnitude in the bins. Looking at neighboring blocks and use them to normalise helps reduce these issues.
6. Concat the hists of the cells for every block to create a feature vector!

In order to deal with rotation and different angles of the object we can use multiple feature vectors and learn a model in order to predict the object. 
I used a SVM for this task, because I have a small amount of training samples and high demensional vector. 

## Implementation

The file structure consists of a folder called "train". This folder is devided into "is" and "not". In the "is" folder you should place the object which you want to detect. In the not folder you want to place some background.
Next you can run "generate_train_data.py" which takes patches out of the object images or background and saves them in "train_preprocessed" in order to generate more training data. Look at what the best settings for the "crop" and "stride" paramters must be.
"HOG.py" has all the main logic of the hog feature extraction. "detection.py" first initializes the HOG dectector, first you need to train the SVM with the training data. If you have already trained the model you can comment this line and just load the model. 
Next it uses an image, you can try my "test_img.jpg" and traverses with a stride over the image and check what the prop is that that patch is the object. It return the best match and draws the location on the gradient and color image. 

Some results:

![alt text](https://github.com/WanneWisse/HogDetection/blob/master/readme2.png?raw=true)
![alt text](https://github.com/WanneWisse/HogDetection/blob/master/readme1.png?raw=true)


Tip: The object which you want to detect should be well defined by gradients. Play around with the threshold to delete noisy gradients with low maginitude. Also check the "detector_output" map to see what the top 20 patches were to get more insight in the performance of the model.

And most of all, have fun!
