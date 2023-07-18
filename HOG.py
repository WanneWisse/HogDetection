import numpy as np
from scipy.signal import convolve2d
import cv2
import math
from joblib import dump, load
import glob
from sklearn import svm

#class to extract hog features and train an SVM based on the HOG features
class HogDetector():
    def __init__(self):
        #kernel to detect the gradient X and Y(X.T) 
        self.perwit_kernel_x = np.array([[1, 0,-1],
                        [2, 0,-2],
                        [1, 0,-1]])
        #threshold for magnitude of gradients, too low gradients cause too much noise
        self.treshold = 150
        #amount of pixels from which a hist is calculated(cell_size*cell_size)
        self.cell_size = 8
        #amount of cells from which a normalization is done(self.block_size*self.block_size) 
        self.block_size = 2
        #the SVM classifier or anyother
        self.clf = None
    
    #function to verify if a patch corresponds to the object by using the clf 
    def detect(self,patch_to_detect):
        #get the featurevector for the patch
        feature_vector_test_image = self.get_feature_vector_given_image(patch_to_detect,self.perwit_kernel_x,self.treshold,self.cell_size,self.block_size)
        #if you dont want to use a SVM you can use underneath formula
        #diff = np.sum(np.abs(self.feature_vector - feature_vector_test_image)
        #return the score for the feature_vector
        return self.clf.predict_proba([feature_vector_test_image])
    
    #load the clf model if saved 
    def load_model(self,path):
        self.clf = load(path) 
    
    #train the SVM using the generate_train_data 
    def train(self):
        #x are the feature vectors
        X = []
        #y are the labels
        y = []

        #for positive classes convert to gray, get feature vector and 1 means positive class
        files = glob.glob('train_preprocessed/is/*')
        for f in files:
            image = cv2.imread(f.title())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.array(image)
            feature_vector = self.get_feature_vector_given_image(image,self.perwit_kernel_x,self.treshold,self.cell_size,self.block_size)
            X.append(feature_vector)
            y.append(1)

        #for negative classes convert to gray, get feature vector and 0 means positive class
        files = glob.glob('train_preprocessed/not/*')
        for f in files:
            image = cv2.imread(f.title())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.array(image)
            feature_vector = self.get_feature_vector_given_image(image,self.perwit_kernel_x,self.treshold,self.cell_size,self.block_size)
            X.append(feature_vector)
            y.append(0)

        #train a support vector machine using the data
        self.clf = svm.SVC(probability=True)        
        self.clf.fit(X, y)
        print(self.clf.classes_)

        #dump the model to a filename
        dump(self.clf, 'model.joblib') 

    #find the gradients in x or y direction using a kernel
    def find_gradients_for_axis(self,image,kernel):
        #Perform convolution
        #same output size and a symetrical boundary
        G_axis = convolve2d(image, kernel, boundary='symm', mode='same')
        return G_axis

    #calculate the magnitude for the x and y gradients and threshold the values to delete noise
    def calculate_magnitude_of_gradients(self,Gx,Gy,treshold):
        Gm = np.sqrt(Gx**2 + Gy**2)
        # Set values smaller than treshold to zero
        Gm[Gm < treshold] = 0
        return Gm

    #determine direction of the gradients using arctan2
    def calculate_direction_of_gradients(self,Gx,Gy):
        Gd = np.arctan2(Gy,Gx) 
        Gd = np.degrees(Gd)
        Gd = (Gd +360)%360
        return Gd

    #create gradient hists from group of pixels(cells)  
    def get_cells(self,Gm,Gd,stride):
        columnsize=Gm.shape[1]
        rowsize=Gm.shape[0]
        #resulting matrix will be size/stride
        result_size = (rowsize//stride,columnsize//stride)
        cells_hist = np.empty(result_size, dtype=object)
        #per cell
        for rows in range(0,rowsize-1,stride):
            for cols in range(0,columnsize-1,stride):
                cellm = Gm[rows:rows+stride,cols:cols+stride]
                celld = Gd[rows:rows+stride,cols:cols+stride]
                #for cell calc hist
                cell_hist = self.calculate_histogram(cellm,celld)
                #set hist on cell place in result matrix
                cells_hist[rows//stride,cols//stride] = cell_hist
        return cells_hist

    #normalise every block of cells using l2 norm
    def normalise_block(self,block_hists):
        total_sum = 0.001
        #concat all hists of cells
        block_hists = block_hists.flatten()
        for hist in block_hists:
            for item in hist:
                total_sum += item**2
        total_sum = math.sqrt(total_sum)

        normalised_block = []
        for hist in block_hists:
            for item in hist:
                normalised_block.append(item/total_sum)
        return normalised_block

   
    # check to which bin[0-360] a direction belongs 
    def calculate_bin(self,direction,binspace):
        si = 0
        sv = abs(direction-binspace[si])
        for bini in range(len(binspace)):
            dist = abs(direction-binspace[bini])
            if dist < sv:
                si = bini
                sv = dist
        return si 
    
    #calculate the histogram for a cell
    def calculate_histogram(self,cellm,celld):
        #9 bins between 0-360
        binspace = np.linspace(0,360,9)
        bins = [0 for binitem in binspace]
        for index, value in np.ndenumerate(cellm):
            maginitude = cellm[index]
            direction = celld[index]
            sbini = self.calculate_bin(direction,binspace)
            #given the bin based on direction add the magnitude
            bins[sbini] += maginitude
        return bins

    #general function to get a feature vector from cell hist matrix
    def get_feature_vector_given_cell_hists(self,cell_hists,block_size):
        columnsize=cell_hists.shape[1]
        rowsize=cell_hists.shape[0]
        feature_vector = []
        for rows in range(0,rowsize-block_size+1):
            for cols in range(0,columnsize-block_size+1):
                #get a block of cells
                block = cell_hists[rows:rows+block_size,cols:cols+block_size]
                #normalise block
                normalised_block = self.normalise_block(block)
                #concat features
                feature_vector += normalised_block
        return feature_vector
    
    #find the gradients in x,y direction, the magintude and the direction of the gradients
    def get_gradients(self,img,kernel,treshold):
        Gx = self.find_gradients_for_axis(img,kernel)
        Gy = self.find_gradients_for_axis(img,kernel.T)
        Gm = self.calculate_magnitude_of_gradients(Gx,Gy,treshold)
        Gd = self.calculate_direction_of_gradients(Gx,Gy)
        return Gx,Gy,Gm,Gd

    #general function to get a feature vector from image
    def get_feature_vector_given_image(self,img,kernel,treshold,cell_size,block_size):
        #step1: gradients of pixels
        Gx,Gy,Gm,Gd = self.get_gradients(img,kernel,treshold)
        #step2: cells of gradients
        cell_hists = self.get_cells(Gm,Gd,cell_size)
        #step3: blocks of cells -> normalise blocks -> concat blocks
        feature_vector = self.get_feature_vector_given_cell_hists(cell_hists,block_size)
        feature_vector = np.array(feature_vector)
        return feature_vector



   