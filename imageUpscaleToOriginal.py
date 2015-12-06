import PIL.Image
import os
import numpy as np

origImageSizes = open("../OriginalImageSizes.txt","r")

data = origImageSizes.readlines()
collectImageDimensions = []
for eachImage in data:
    collectImageDimensions.append(eachImage[:-1].split('\t'))
    #collectImageDimensions = collectImageDimensions + eachImage[:-1].split('\t')

#TODO: use this to find the array for the image in order to get dimensions
original = collectImageDimensions[np.where(np.array(collectImageDimensions) == 'im01986.jpg')[0][0]]

#TODO: put data into an image the size of the predicted binary matrix
#TODO: that is im.putdata([0,150,0,150,0,150,0,150,0,150,0,150,0,150,0,150,0,150,0,150,0,150,0,150])
#TODO: upscale the binary matrix 
#TODO: that is binaryPredicted.resize((original[1],original[2]),Image.ANTIALIAS), may not be just black and white
#TODO: convert upscaled image to black and white
#TODO: that is backTOriginImBW = im.point(lambda x: 0 if x<128 else 255, '1')
#TODO: save bw image into appropriate folder to give to TA (test on this)
#TODO: that is
#TODO: bw.save("../testGrey4.png","PNG")

#http://www.pythonware.com/media/data/pil-handbook.pdf
#http://stackoverflow.com/questions/2111150/create-a-grayscale-image
#http://www.cs.uregina.ca/Links/class-info/325/PythonPictures/#Grayscale
##backToOrigImBW = PIL.Image.new('L', (original[1],original[2])) #L means 8bit black or white
##im.putdata([0,150,0,150,0,150,0,150,0,150,0,150,0,150,0,150,0,150,0,150,0,150,0,150])
##bw = im.point(lambda x: 0 if x<128 else 255, '1')
##
