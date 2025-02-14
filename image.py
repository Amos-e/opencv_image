#Import opencv
import cv2 as cv
import numpy as np
import sys

try:
    if sys.argv[1]:

        #Read the image
        image_path = sys.argv[1]
        img = cv.imread(image_path)

        #Resize the image
        img = cv.resize(img, (500,600))

        #Create an ndarray with numpy of rows and columns matching with the 
        #dimensions of the image 
        image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        black_and_white = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        #Open the file to which we'll write the image matrix
        file = open("image_matrix.txt", 'w')
        file.writelines("[\n")

        #Indices for the 2 dimensional matrix for the gray scale image
        index0 = 0
        index1 = 0

        for row in img:
            file.writelines("[")
            for cell in row:
                blue, green, red = cell[0], cell[1], cell[2]
                gray_scale = (0.3 * red) + (0.59 * green) + (0.11 * blue)

                file.writelines(f'[{blue} {green} {red}]')

                image[index0, index1] = gray_scale
                black_and_white[index0, index1] = 0 if int(gray_scale)  < 128 else 255
                index1 += 1

            file.writelines("]\n\n")

            index0+=1
            index1=0

        file.writelines("]\n")
        file.close()

        #Face Detection
        f_cascade = cv.CascadeClassifier("face.xml")
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        face = f_cascade.detectMultiScale(gray_image, 1.3, 5)

        x1, y1, width, height = face[0]
        print(face[0])

        region = gray_image[y1:y1+height, x1:x1+width]
        blurred_roi = cv.GaussianBlur(region, (29,29), 0)
        gray_image[y1:y1+height, x1:x1+width] = blurred_roi

        #Gray Scale Image
        cv.imshow("Gray Scale", image)
        cv.imwrite("gray_scale.png", image)

        # #Black and White Image
        cv.imshow("Black and White", black_and_white)
        cv.imwrite("black_and_white.png", black_and_white)

        #Blurred Face
        cv.imshow("Blurred Face", gray_image)
        cv.imwrite("blurred_face", gray_image)

        cv.waitKey(0)
        cv.destroyAllWindows()

except IndexError:
    print("No Image Selected")