import sys
import os

from pyspark import SparkConf, SparkContext

def processImage(imageName):

    import numpy as np
    import math
    import PIL.Image

    imagePath  = "/scratch/tosadojimenez/images/bdagroup5/Original/trainAndVal/"+imageName
    skinPath = "/scratch/tosadojimenez/images/bdagroup5/Skin/trainAndVal/"+imageName[0:-4]+"_s.bmp"
    skin = PIL.Image.open(skinPath)
    image = PIL.Image.open(imagePath)
    pixels = image.load()
    skinPixels = skin.load()                          
    blockSideSize = 7
    imSizeX,imSizeY = image.size
    imSizeRGB = 3

    fringePixels = np.int(math.floor(blockSideSize/2)) # to be ignored, use sizeOfBlock x sizeOfBlock blocks
    columnSampleSize = imSizeY - fringePixels*2
    rowSampleSize = imSizeX - fringePixels*2
    numOfSamples=(imSizeX-fringePixels*2)*(columnSampleSize) #eg. if x side of image(and y is 7) is 10 then only have 4 samples 

    numOfColsPerPartialSample =  blockSideSize*blockSideSize*3
    partialSample = np.zeros((1,numOfColsPerPartialSample),'uint8')
    partialSamples = np.zeros((numOfSamples,numOfColsPerPartialSample),'uint8')

    numOfColumnsPerSample = numOfColsPerPartialSample+1 # the cube made by the block and rbg + 1 for class label of skin
    sample = np.zeros((1,numOfColumnsPerSample),'uint8')
    samples = np.zeros((numOfSamples,numOfColumnsPerSample),'uint8')

    isSkin = 1
    currentSample=0

    for x in  range(0+fringePixels,imSizeX-fringePixels):
        for y in range(0+fringePixels,imSizeY-fringePixels):
            if ((np.array(skinPixels[x,y])==255).all()):
                    isSkin=np.uint8(0)
            partialSampleIndex = 0
            for blockFringe in range(fringePixels, 0,-1): #use fringes in order have the spiral order, skip if fringe is 0                      
                xBlock = x-blockFringe   # top section of outer block
            for yBlock in range(y-blockFringe,y+blockFringe+1):#+1 because it excludes last index
                partialSample[0,partialSampleIndex:partialSampleIndex+imSizeRGB] = np.array(pixels[xBlock,yBlock])                        
                partialSampleIndex =partialSampleIndex+imSizeRGB #next 3 RGB values
            yBlock = y+blockFringe #middle section right 
            for xBlock in range(x-(blockFringe-1),x+blockFringe):
                partialSample[0,partialSampleIndex:partialSampleIndex+imSizeRGB] = np.array(pixels[xBlock,yBlock])
                partialSampleIndex = partialSampleIndex+imSizeRGB
            yBlock = y-blockFringe  #middle section left
            for xBlock in range(x-(blockFringe-1),x+blockFringe):
                partialSample[0,partialSampleIndex:partialSampleIndex+imSizeRGB] = np.array(pixels[xBlock,yBlock])
                partialSampleIndex = partialSampleIndex+imSizeRGB
            xBlock = x+blockFringe #bottom section
            for yBlock in range(y-blockFringe,y+blockFringe+1):
                partialSample[0,partialSampleIndex:partialSampleIndex+imSizeRGB] = np.array(pixels[xBlock,yBlock])
                partialSampleIndex = partialSampleIndex+imSizeRGB
            #for the middle point (same as when block side size is 1)
            partialSample[0,partialSampleIndex:partialSampleIndex+imSizeRGB] = np.array(pixels[x,y]) 
            sample[0,0:numOfColsPerPartialSample]=partialSample
            sample[0,numOfColumnsPerSample-1] = isSkin
            samples[currentSample,:]= sample
            currentSample=currentSample+1;
            isSkin=1;
    
    outputSamples = np.array(samples)
    return (outputSamples)
    #np.savetxt("/scratch/tosadojimenez/images/bdagroup5/Preprocessed/samples_"+imageName[0:-4]+"_bs"+str(blockSideSize)+".txt",samples,fmt='%d')

#if __name__ == '__main__':
        
    #toParallel = os.listdir("/scratch/tosadojimenez/images/bdagroup5/Original/trainAndVal/")
    #parallelFiles = sc.parallelize(toParallel[0:2])
    #parallelFiles.map(processImage)
