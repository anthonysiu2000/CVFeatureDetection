import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

import transformations

## Helper functions ############################################################

def inbounds(shape, indices):
    '''
        Input:
            shape -- int tuple containing the shape of the array
            indices --  int list containing the indices we are trying 
                        to access within the array
        Output:
            True/False, depending on whether the indices are within the bounds of 
            the array with the given shape
    '''
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    # Implement in child classes
    def detectKeypoints(self, image):
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
        Compute silly example features. This doesn't do anything meaningful, but
        may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage --  numpy array containing the Harris score at
                            each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'. Also compute an 
        # orientation for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN
        
        #creates a gaussian mask with 0.5 sigma of size 5 by 5
        #initializes mask as 5x5 with center = 1
        mask = np.zeros((5,5))
        mask[2][2] = 1
        #applies gaussian blur to mask
        gaussMask = ndimage.gaussian_filter(mask, 0.5)
        
        #creates a 3x3 sobel filters for x and y derivatives
        empty = np.zeros((3,3))
        empty[1][1] = 1
        xSobel = ndimage.sobel(empty, axis = 1)
        ySobel = ndimage.sobel(empty, axis = 0)
        
        
        #Compute the Harris Matrix for the image
        
        #first gets the 1st order derivative for x and y
        gradx = ndimage.convolve(srcImage, xSobel, mode = 'nearest')
        grady = ndimage.convolve(srcImage, ySobel, mode = 'nearest')
        
        # #gets the 2nd order derivative for x twice
        # gradxx = ndimage.convolve(gradx, xSobel, mode = 'nearest')
        # #applies the gaussian mask to the 2nd order derivative
        # Gradxx = ndimage.convolve(gradxx, gaussMask, mode = 'nearest')
        
        gradxx = gradx * gradx
        Gradxx = ndimage.convolve(gradxx, gaussMask, mode = 'nearest')
        
        # #gets the 2nd order derivative for y twice
        # gradyy = ndimage.convolve(grady, ySobel, mode = 'nearest')
        # #applies the gaussian mask to the 2nd order derivative
        # Gradyy = ndimage.convolve(gradyy, gaussMask, mode = 'nearest')
        
        gradyy = grady * grady
        Gradyy = ndimage.convolve(gradyy, gaussMask, mode = 'nearest')
        
        # #gets the 2nd order derivative for both x and y
        # gradxy = ndimage.convolve(gradx, ySobel, mode = 'nearest')
        # #applies the gaussian mask to the 2nd order derivative
        # Gradxy = ndimage.convolve(gradxy, gaussMask, mode = 'nearest')
        
        gradxy = gradx * grady
        Gradxy = ndimage.convolve(gradxy, gaussMask, mode = 'nearest')
        
        for i in range(height):
            for j in range(width):
                #Calculates the Harris corner strength matrix
                #c(H) = det(H) − 0.1(trace(H))^2
                harrisImage[i][j] = (Gradxx[i][j] * Gradyy[i][j] - Gradxy[i][j]**2) - 0.1 * (Gradxx[i][j] + Gradyy[i][j]) ** 2
                #finds the orientation of the gradient
                #math atan2 does arctan with respect to the signs of x and y, so we arent limited to pi/2 to -pi/2
                angleRad = math.atan2(grady[i][j], gradx[i][j])
                #convets to deg
                orientationImage[i][j] = angleRad * 180 / math.pi
        
        # TODO-BLOCK-END

        return harrisImage, orientationImage





    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                            each pixel.
        Output:
            destImage --numpy array containing True/False at
                        each pixel, depending on whether
                        the pixel value is the local maxima in
                        its 7x7 neighborhood.
        '''
        destImage = np.zeros_like(harrisImage, np.bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        
        
        #brute force method to check the 7x7 neighborhood of each pixel is <= the current pixel's score
        
        # h = destImage.shape[0]
        # w = destImage.shape[1]
        
        # #loop through every pixel
        # for i in range(h):
        #     for j in range(w):
        #         #loops through the 7x7 neighborhood
        #         for a in range(-3,4):
        #             nestedbreak = False
        #             for b in range(-3,4):
        #                 #skips if neighbor is out of bounds:
        #                 if i+a < 0 or i+a >= h or j+b < 0 or j+b >= w:
        #                     continue
        #                 #if neighbor is greater than current pixel, this pixel is not a local maxima
        #                 if harrisImage[i+a][j+b] > harrisImage[i][j]:
        #                     destImage[i][j] = False
        #                     nestedbreak = True
        #                     break
        #             if nestedbreak:
        #                 break
        
        
        #applies a 7x7 max filter to the harris image
        #mode = nearest is not needed as we are taking the maxima of the 7x7 neighborhood
        localMax = ndimage.maximum_filter(harrisImage, size = 7)
        
        #if the localMax pixel is equal to the image pixel, then that pixel is a local maxima in the image
        destImage = np.equal(harrisImage, localMax)
        
        
        # TODO-BLOCK-END

        return destImage










    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.
        # You will need to implement this function.
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                # TODO-BLOCK-BEGIN
                
                f.size = 10
                f.pt = (x, y)
                f.angle = orientationImage[y][x]
                f.response = harrisImage[y][x]
                
                # TODO-BLOCK-END

                features.append(f)

        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        detector = cv2.ORB_create()
        return detector.detect(image)




## Feature descriptors #########################################################

class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints --the detected features, we have to compute the feature
                        descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        for i, f in enumerate(keypoints):
            x, y = int(f.pt[0]), int(f.pt[1])

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            # Note: use grayImage to compute features on, not the input image
            # TODO-BLOCK-BEGIN
            
            #initialize this feature descriptor as a 25 element list
            tempDesc = []
            
            #loops through the 5x5 window in row major order
            for j in range(-2, 3):
                for k in range(-2, 3):
                    #if pixel is out of bounds, append a 0
                    if x+k < 0 or x+k >= grayImage.shape[1] or y+j < 0 or y+j >= grayImage.shape[0]:
                        tempDesc.append(0)
                    else: #append the pixel value
                        tempDesc.append(grayImage[y+j][x+k])
            
            #convert to numpy and add to the descriptor
            desc[i] = np.array(tempDesc)
            
            # TODO-BLOCK-END

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            transMx = np.zeros((2, 3))

            # TODO 5: Compute the transform as described by the feature
            # location/orientation and store in 'transMx.' You will need
            # to compute the transform from each pixel in the 40x40 rotated
            # window surrounding the feature to the appropriate pixels in
            # the 8x8 feature descriptor image. 'transformations.py' has
            # helper functions that might be useful
            # Note: use grayImage to compute features on, not the input image
            # TODO-BLOCK-BEGIN
            
            x, y = int(f.pt[0]), int(f.pt[1])
            
            #The Transition matrix is the matrix product of 4 transformations
            #Translation, Rotation, Scaling, and Translation
            
            #Matrix to translate to the center of the 40x40 window
            translateM = transformations.get_trans_mx(np.array([-x,-y, 0]))
            
            #Matrix to rotate the 40x40 window to orientation of 0 degrees
            toRotate = np.radians(-f.angle)
            rotateM = transformations.get_rot_mx(0, 0, toRotate)
            
            #Matrix to scale the 40x40 window to 8x8 aka a 0.2 scale factor
            scaleM = transformations.get_scale_mx(0.2, 0.2, 1)
            
            #matrix to translate to the corner of the 8x8 window
            translateM2 = transformations.get_trans_mx(np.array([4, 4, 0]))
            
            #perform matrix multiplication
            
            temp = np.matmul(translateM2, scaleM)
            temp = np.matmul(temp, rotateM)
            temp = np.matmul(temp, translateM)
            
            #obtains the important values to put into the 2x3 matrix
            
            #x and y rotation
            transMx[0][0] = temp[0][0]
            transMx[0][1] = temp[0][1]
            transMx[1][0] = temp[1][0]
            transMx[1][1] = temp[1][1]
            #x and y translation
            transMx[0][2] = temp[0][3]
            transMx[1][2] = temp[1][3]
            
            # TODO-BLOCK-END
            
            # if transMx[0][0] != 0.2:
            #     print(temp)
            #     print(transMx)
            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx,(windowSize, windowSize), flags=cv2.INTER_LINEAR)



            # TODO 6: Normalize the descriptor to have zero mean and unit
            # variance. If the variance is negligibly small (which we
            # define as less than 1e-10) then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN
            # if transMx[0][0] != 0.2:
            #     print(destImage)
            #obtain mean of the descriptor
            mean = np.mean(destImage)
            destImage = destImage - mean
            
            #obtain variance of the descriptor
            var = np.var(destImage)
            
            #if variance less than 1e-10, set descriptor to 0
            if var < 10**(-10):
                destImage = np.zeros((8, 8))
            else:
                destImage = destImage / np.sqrt(var)
            
            desc[i] = destImage.flatten()
            
            
            # TODO-BLOCK-END

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc




## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        raise NotImplementedError

    # Evaluate a match using a ground truth homography. This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        assert desc1.ndim == 2
        assert desc2.ndim == 2
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching. This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN
        #loops through all the features in the first image
        for i,feature in enumerate(desc1):
            #running min ssd
            #running min ssd index
            ssd = np.inf
            ssdI = 0
            #loops through all the features in the second image
            for j,feature2 in enumerate(desc2):
                #calculates ssd between these 2
                tempssd = np.sum((feature - feature2)**2)
                
                #if less than ssd, set ssd to tempssd and index properly
                if tempssd < ssd:
                    ssd = tempssd
                    ssdI = j
            #create a match object
            match = cv2.DMatch()
            match.queryIdx = i
            match.trainIdx = ssdI
            match.distance = ssd
            matches.append(match)
        # TODO-BLOCK-END

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        assert desc1.ndim == 2
        assert desc2.ndim == 2
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image. If the SSD distance is negligibly small, in this case less
        # than 1e-5, then set the distance to 1. If there are less than two features,
        # set the distance to 0.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN
        
        #loops through all the features in the first image
        for i,feature in enumerate(desc1):
            #running min ssd and 2nd min ssd
            #running min ssd index 
            ssd1, ssd2 = np.inf, np.inf
            ssdI = 0
            
            #if there is less than 2 features (only one feature basically)
            if desc2.shape[0] < 2:
                #create a match object
                match = cv2.DMatch()
                match.queryIdx = 0
                match.trainIdx = 0
                match.distance = 0
                matches.append(match)
                break
            
            
            #loops through all the features in the second image
            for j,feature2 in enumerate(desc2):
                #calculates ssd between these 2
                tempssd = np.sum((feature - feature2)**2)
                #if distance is negligibly small, set to 1
                if tempssd < 1e-5:
                    tempssd = 1
                
                #if less than ssd, set ssd to tempssd and index properly
                if tempssd < ssd2:
                    
                    if tempssd < ssd1:
                        ssd2 = ssd1
                        ssd1 = tempssd
                        ssdI = j
                    else:
                        ssd2 = tempssd
                    
                    
            #create a match object
            match = cv2.DMatch()
            match.queryIdx = i
            match.trainIdx = ssdI
            match.distance = ssd1/ssd2
            matches.append(match)
            
            
        # TODO-BLOCK-END

        return matches