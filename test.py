import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial
gauss = cv2.getGaussianKernel(5, 0.5)
mask = np.zeros((5,5))
mask[2][2] = 1

mask = ndimage.gaussian_filter(mask, 0.5)
# print(mask)
mask = np.zeros((3,3))
mask[1][1] = 1
# print(ndimage.sobel(mask, axis = 1, mode = 'nearest'))

image = np.random.rand(5,5)


empty = np.zeros((3,3))
empty[1][1] = 1
xSobel = ndimage.sobel(empty, axis = 1)
ySobel = ndimage.sobel(empty, axis = 0)
#and for 2nd order derivatives

xxSobel = ndimage.sobel(xSobel, axis = 1)
# print(xxSobel)
Gradxx = ndimage.convolve(image, xxSobel, mode = 'nearest')
# print(Gradxx)

# print(image)
gradx = ndimage.convolve(image, xSobel, mode = 'nearest')
# print(gradx)
gradxx = ndimage.convolve(gradx, xSobel, mode = 'nearest')



hello = np.asarray([[1,2,3],[4,5,6],[7,8,9]])
# print(hello)
mask = np.asarray([[1,2,1]])
# mask[1][1] = 1
# mask[1][0] = 1
# mask[1][2] = 1
# mask[0][1] = 1
# mask[2][1] = 1

# print(mask)

convolve = ndimage.convolve(image, mask)
correlate = ndimage.correlate(image, mask)
# print(convolve)
# print(correlate)


K = np.array([1/16, 4/16, 6/16, 4/16, 1/16], dtype=np.float32)
print(K)