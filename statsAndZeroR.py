
import PIL.Image
import os
import numpy as np

numOfSkin = float(0)
totalSkin = float(0)
totalNumOfPixels = float(0)
imageNames = os.listdir("\\cygwin64\\home\\joeltosado\\bdagroup5\\Skin\\train")

resizeStart = 0.1 #default start of resize
endResizeExclusive = 1.1
incrementResizeStep = 0.1
resizeRange = np.arange(resizeStart,endResizeExclusive,incrementResizeStep)
resizeArray = [[0]*3 for _ in range(len(resizeRange))] #2d list
for imageName in imageNames:
    image = PIL.Image.open("\\cygwin64\\home\\joeltosado\\bdagroup5\\Skin\\train\\"+imageName)
    originalWidth, originalHeight = image.size
    totalNumOfPixels = totalNumOfPixels + float(originalWidth)*float(originalHeight)
    copyImage = image.copy()
    skinPixels = copyImage.load()     
    for x in range(originalWidth):
        for y in range(originalHeight):
            if((np.array(skinPixels[x,y])!=255).any()):
                numOfSkin=numOfSkin+1
    totalSkin = totalSkin + numOfSkin
    numOfSkin=float(0) #next
    
    resizeIndex = 0
    print imageName
    for resize in resizeRange:
        copyImage = image.copy()
        if resize != 1 :
            width = int(resize*originalWidth)
            height = int(resize*originalHeight)
            resizedSize= width, height
            copyImage.thumbnail(resizedSize,PIL.Image.ANTIALIAS)
            width,height = copyImage.size
        else:
            width = originalWidth
            height = originalHeight
            
        copyImage.close()
        width = float(width)
        height = float(height)
        #print resize, width,height
        resizeArray[resizeIndex][0] = resizeArray[resizeIndex][0] + (width-2*2)*(height-2*2) #5x5
        resizeArray[resizeIndex][1] = resizeArray[resizeIndex][1] + (width-3*2)*(height-3*2) #7x7
        resizeArray[resizeIndex][2] = resizeArray[resizeIndex][2] + (width-4*2)*(height-4*2) #9x9
        #print imageName + "\t NUMOFSKIN%: "+str((numOfSkin*float(100))/(width*height))+ "\t FRINGE2SAMPLES: " + str((width-2*2)*(height-2*2))+ "\t FRINGE3SAMPLES: " + str((width-3*2)*(height-3*2))+ "\t FRINGE4SAMPLES: " + str((width-4*2)*(height-4*2))
        resizeIndex = resizeIndex + 1
print "totalNumOfSkin%\t" + str((100*totalSkin)/totalNumOfPixels)       
print "ResizeTo:\t5x5Block:\t7x7Block:\t9x9Block"
for resizeInd in range(len(resizeRange)):
    print str(resizeRange[resizeInd])+"\t"+str(resizeArray[resizeInd][0])+"\t"+str(resizeArray[resizeInd][1])+"\t"+str(resizeArray[resizeInd][2])
   
    
##print "ResizeTo:\t"+str(resizeRange)
##print "#Samples:\t"
##    print "resizeImageTo: "+str(resize) +"5x5 Block Samples: "+str(samplesFringeTwo) + ", total number of 3fringe samples: "+str(samplesFringeThree) + ", total number of 4fringe samples: "+str(samplesFringeFour) + ", total skin pixels %: " + str((totalSkin*100)/totalNumOfPixels)            
