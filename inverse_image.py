import cv2 as cv
import numpy as np

image_path = 'my_image.jpeg'

img = cv.imread(image_path, 0)
img = cv.resize(img, (500,500))

#Image Inverse
inverse = np.linalg.inv(img)

#Identity matrix for image
identity = np.multiply(img, inverse)

#Transpose 1
transpose = img.transpose()

#Transpose 2
transpose2 = transpose.transpose()

#Display Images
cv.imshow("Inverse Image", inverse)
cv.imshow("Identity Image", identity)
cv.imshow("Transpose 1", transpose)
cv.imshow("Transpose 2", transpose2)

cv.waitKey(0)
cv.destroyAllWindows()