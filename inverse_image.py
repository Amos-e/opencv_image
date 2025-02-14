import cv2 as cv
import numpy as np

image_path = 'my_image.py'

img = cv.imread(image_path,0)
img = cv.resize(img, (500,500))

#Image Inverse
inverse = np.linalg.inv(img)

#Identity matrix for image
identity = np.multiply(img, inverse)

transpose = img.transpose()
transpose2 = transpose.transpose()


cv.imshow("Image", transpose2)
cv.waitKey(0)
cv.destroyAllWindows()