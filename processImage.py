def processImage(imageName, blockSideSize=7, resizeTo = None, test = 0, val=0):
    '''
        Update on 12/05/15: Change the resize feature to be a ratio
    '''

    import numpy as np
    import math
    import PIL.Image 
    
    if val == 1:
        imagePath  = "../Original/val/"+imageName
        skinPath = "../Skin/val/"+imageName[0:-4]+"_s.bmp"
    else:
        imagePath  = "../Original/train/"+imageName
        skinPath = "../Skin/train/"+imageName[0:-4]+"_s.bmp"

    skin = PIL.Image.open(skinPath)
    image = PIL.Image.open(imagePath)
    imSizeX,imSizeY = image.size

    if resizeTo !=  None:
        imSizeX = int(resizeTo*imSizeX)
        imSizeY = int(resizeTo*imSizeY)
        resizedSize= imSizeX, imSizeY
        image.thumbnail(resizedSize,PIL.Image.ANTIALIAS)
        skin.thumbnail(resizedSize, PIL.Image.ANTIALIAS)
        imSizeX,imSizeY = image.size
    
    pixels = image.load()
    skinPixels = skin.load()                          
    imSizeRGB = 3

    fringePixels = np.int(math.floor(blockSideSize/2)) # to be ignored, use sizeOfBlock x sizeOfBlock blocks
    columnSampleSize = imSizeY - fringePixels*2
    rowSampleSize = imSizeX - fringePixels*2
    numOfSamples=(imSizeX-fringePixels*2)*(columnSampleSize) #eg. if x side of image(and y is 7) is 10 then only have 4 samples 

    numOfColsPerPartialSample =  blockSideSize*blockSideSize*3
    partialSample = np.zeros((1,numOfColsPerPartialSample))
    partialSamples = np.zeros((numOfSamples,numOfColsPerPartialSample))

    numOfColumnsPerSample = numOfColsPerPartialSample+1 # the cube made by the block and rbg + 1 for class label of skin
    sample = np.zeros((1,numOfColumnsPerSample))
    samples = np.zeros((numOfSamples,numOfColumnsPerSample))

    isSkin = 1
    currentSample=0

    for y in range(0+fringePixels,imSizeY-fringePixels):
        for x in  range(0+fringePixels,imSizeX-fringePixels):
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
            partialSample[0,partialSampleIndex:partialSampleIndex+imSizeRGB] = np.array(pixels[x,y])#for the middle point (same as when block side size is 1) 
            sample[0,0:numOfColsPerPartialSample]=partialSample
            if test==1:
                partialSamples[currentSample,:] = partialSample
            else:
                sample[0,numOfColumnsPerSample-1] = isSkin
                samples[currentSample,:]= sample
                isSkin=1;
            currentSample=currentSample+1;
            
    if test==1:
        samples = partialSamples
    
    return (np.asarray(samples, dtype=np.uint8))
