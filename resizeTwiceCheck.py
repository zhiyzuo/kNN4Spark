import PIL.Image
import os
import math

images = os.listdir("../Original/val/")
countRight = 0
try1 = 0

for eachImage in images:
    print "start\t"+eachImage
    originalImage = PIL.Image.open("../Original/val/"+eachImage)
    originalWidth, originalHeight = originalImage.size
    copyImage = originalImage.resize((originalWidth*0.75,originalHeight*0.75),PIL.Image.ANTIALIAS)
    copyWidth,copyHeight = copyImage.size
    if(try1 == 0):
        copyImage.save("../Original/resizeTest"+eachImage+".jpg","JPEG")
        try1=1
    if (math.ceil(copyWidth*1/0.75) == originalWidth)& (math.ceil(copyHeight*1/0.75)==originalHeight):
        countRight=countRight+1
    else:
        print "copyWidth="+str(copyWidth)+","+ str(copyWidth*1/0.75)+",originalWidth="+str(originalWidth)+"; copyHeight="+str(copyHeight)+","+str(copyHeight*1/0.75)+",originalHeight="+str(originalHeight)
    

if countRight == len(images):
    print "shitz workin"
