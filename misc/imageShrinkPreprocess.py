import PIL.Image
import os
import numpy as np

#different images paths with images to shrink. Assumes that it exists
imagePaths = list(["/Original/val/","/Original/train/","/Skin/val/","/Skin/train/"])

#resizings for images
resizeRange = np.arange(0.1,1.1,0.1)

#create folder structure for where the resized images will reside
for imagePath in imagePaths:
    for resize in resizeRange:
        resizedPath = "../"+str(int(resize*10))+imagePath
        if not os.path.exists(resizedPath):
            os.makedirs(resizedPath)

# to output the original sizes since resize does truncating
# and thumbnail can give other values than the ones
# provided(fixes one) when resizing
textFile = open("../OriginalImageSizes.txt","w")

originalCount = 0

#deposit shrunk images in this directory
for imagePath in imagePaths:
    imageSubset = os.listdir(".."+imagePath)
    for imageName in imageSubset:   
        imageOriginal = PIL.Image.open(".."+imagePath+imageName)
        originalWidth, originalHeight = imageOriginal.size
        textFile.write("{0}\t{1}\t{2}\n".format(imageName,originalWidth,originalHeight))
        print imageName
        for resize in resizeRange:
            imageCopy = imageOriginal.copy()
            if resize==1.0:
                imageCopy = imageOriginal
                width = originalWidth
                height = originalHeight
            else:
                imageCopy.thumbnail((originalWidth*resize,originalHeight*resize),PIL.Image.ANTIALIAS)
                width, height = imageCopy.size
            resizedImagePath = "../"+str(int(resize*10))+imagePath+imageName
            if originalCount < 2:
                imageCopy.save(resizedImagePath,"JPEG") 
            else:
                imageCopy.save(resizedImagePath,"BMP")
    originalCount = originalCount + 1            
textFile.close()
