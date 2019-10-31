"""A module to read FEI SER files from TIA. The reader can handle 1D and 2D data sets such as
images, sets of images, spectra, and sets of spectra (maps and line scans). The output is
a python dictionary with the data and some metadata.
usage:
  data = serReader.serReader('filename.ser')
usage tips:
 - To see all available variables use the command data.keys()
 - Image data is contained in data['imageData']
 - Spectra data is contained in data['spectra']
Information about multidimensional datasets is contained in the 'scan...' dictionary keys.
"""
 
import numpy as np
#import scipy as sp

def getType(dataType):
    """Return the correct data type according to TIA's TYPE list"""
    if dataType == 1:
        Type = np.uint8
    elif dataType == 2:
        Type = np.uint16
    elif dataType == 3:
        Type = np.uint32 #uint32
    elif dataType == 4:
        Type = np.int8 #int8
    elif dataType == 5:
        Type = np.int16 #int16
    elif dataType ==  6:
        Type = np.int32 #int32
    elif dataType ==  7:
        Type = np.float32 #float32
    elif dataType ==  8:
        Type = np.float64 #float64
    elif dataType == 12:
        Type = np.int32 #int32 4 byte signed integer (same as dataType==6?)
    else:
        print("Unsupported data type: "+str(dataType)) #complex data types are currently unsupported
        Type = -1
    return Type
#END getType()

def serReader(fname):
    """ Reads in data in FEI's TIA .SER format. 1D (Spectra) and 2D (images) formats are both supported.
    Data is returned in a dict. 
    
    Paremeters 
    ------
    fname : string
        The file name of the SER file to load. Include the full path if the file is not in the current working directory
    
    Returns
    ------
    dataOut : dict
        A Python dict containing the spectral or image data in the SER file. This includes meta data such as the pixel sizes.
        
    Examples
    -------
    >>> from file_readers import serReader
    >>> im1 = serReader.serReader('image1_1.ser') #read in the data from the file
    2D dataset with: 1 image(s)
    >>> im1.keys() #show all information loaded from the file
    dict_keys(['pixelSizeY', 'scanDescription', 'pixelSizeX', 'xAxis', 'scanUnit', 'imageData', 'filename', 'scanCalibration', 'yAxis'])
    >>> im1['pixelSizeX'] #print out the X axis pixel size in meters
    9.5708767988960588e-11
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(im1['imageData'],cmap='gray',origin='lower',aspect='equal') #Show the data as an image as it is displayed in TIA
    <matplotlib.image.AxesImage at 0x217d6c1dcf8>
    """
    #import sys, math
    #import array
    
    f = open(fname,'rb')

    #0 byteorder, 1 seriesID, 2 seriesVersion
    head1 = np.fromfile(f,dtype=np.int16,count=3)
    
    TIASeriesVersion = head1[2]    
    if TIASeriesVersion <= 528:     
        offsetArrayType = np.int32 #used for older versions of TIA before TIA 4.7.3 (Titan 1.6.4)
    else:
        offsetArrayType = np.int64 #used for older versions of TIA after TIA 4.7.3 (Titan 1.6.4)
    
    #0 datatypeID, 1 tagtypeID, 2 totalNumberElements
    #3 validNumberElements, 4 offsetArrayOffset, 5 numberDimensions
    head2 = np.fromfile(f,dtype=np.int32,count=6)
    offsetArrayOffset = head2[4] #The byte offset to the data and tag offset arrays
    validNumberElements = head2[3] #The valid number of elements to read
    numberDimensions = head2[5] #The number of dimensions in the scanning directions (as in a spectral map)
    
    dimSize = np.zeros((numberDimensions,),dtype=np.int32)
    dimCalibration = np.zeros((2,numberDimensions),dtype=np.float64)
    dimCalElement =  np.zeros((numberDimensions,),dtype=np.int32)
    dimDescription = np.zeros((numberDimensions,),dtype=np.int32)
    
    dimensionDescription = ''
    scanUnit = ''
    #Read in the dimension arrays
    for kk in range(0,numberDimensions):
        dimSize[kk] = np.fromfile(f,dtype=np.int32,count=1) #number of scanning pixels along this dimension
        dimCalibration[:,kk] = np.fromfile(f,dtype=np.float64,count=2) #calibration [0] offset and [1] delta and [2] element
        dimCalElement = np.fromfile(f,dtype=np.int32,count=1) #element at which the calibration offset should be applied
        dimDescription[kk] = np.fromfile(f,dtype=np.int32,count=1) #dimension description       
        #print(dimDescription[kk])
        #Describes the dimension (e.g. Number)
        dimensionDescription = np.fromfile(f,dtype=np.int8,count=dimDescription[kk]).tostring()
        #print(dimensionDescription)
        #9 unitsLength
        unitsLength = np.fromfile(f,dtype=np.int32,count=1) #this is read as an array but needs to be just an integer. Use unitsLength[0] to remove the array part
        
        #unit string (e.g. meters)
        readUnit = np.fromfile(f,dtype=np.int8,count=unitsLength[0])
        scanUnit = readUnit.tostring() #the units string (e.g. meters)
    
    f.seek(offsetArrayOffset,0) #move to the start of the offset arrays
    dataOffsetArray = np.fromfile(f,dtype=offsetArrayType,count=head2[3]) #offsetArrayType is int64 for newer TIA versions (>4.7.3 and maybe earlier)
    tagOffsetArray = np.fromfile(f,dtype=offsetArrayType,count=head2[3])
        
    #1D data elements datatypeID is hex: 0x4120 or decimal 16672
    if head2[0] == 16672:
        print("1D datasets (spectra) detected. Careful, not fully tested")
        
        for ii in range(0,validNumberElements):
            f.seek(dataOffsetArray[ii],0) #seek to the correct byte offset
            
            #read in the data calibration information
            calibration = np.fromfile(f,dtype=np.float64,count=2)
            calibrationElement = np.fromfile(f,dtype=np.int32,count=1)
            dataType = np.fromfile(f,dtype=np.int16,count=1)
            Type = getType(dataType) #convert SER data type to Python data type string
            arrayLength = np.fromfile(f,dtype=np.int32,count=1)
            arrayLength = arrayLength[0] #change from array to integer
            dataValues = np.fromfile(f,dtype=Type,count=arrayLength)
            
            #Create multidimensional spectra array on first pass
            if ii == 0:
                spectra = np.zeros((validNumberElements,arrayLength))

            spectra[ii,:] = dataValues
            
        #Add calibrated data to start of spectra array
        eLoss = np.linspace(calibration[0],calibration[0] + (arrayLength-1)*calibration[1],arrayLength)
        
        print("Spectra information: Offset = %f, Delta = %f" % (calibration[0], calibration[1]))
        
        if numberDimensions > 1:
            spectra = spectra.reshape(dimSize[0],dimSize[1],arrayLength)
        
        dataOut = {'spectra':spectra,'eLoss':eLoss,'eOffset':calibration[0],'eDelta':calibration[1],'scanCalibration':dimCalibration,'scanDescription':dimensionDescription,'scanUnit':scanUnit}
        
        return dataOut
            
    #2D data elements have datatypeID as hex: 0x4122 or decimal 16674
    if head2[0] == 16674:
        print("2D dataset with: " + str(head2[3]) + " image(s)")
        for jj in range(0,head2[3]):
            f.seek(dataOffsetArray[jj],0) #seek to start of data
            
            #Read calibration data for X-axis
            calibrationX = np.fromfile(f,dtype=np.float64,count=2) #reads the offset calibrationX[0] and the pixel size calibrationX[1]
            calibrationElementX = np.fromfile(f,dtype=np.int32,count=1)

            #Read calibration data for Y-axis
            calibrationY = np.fromfile(f,dtype=np.float64,count=2)
            calibrationElementY = np.fromfile(f,dtype=np.int32,count=1)

            dataType = np.fromfile(f,dtype=np.int16,count=1)

            Type = getType(dataType[0])
            
            arraySize = np.fromfile(f,dtype=np.int32,count=2)
            arraySizeX = arraySize[0]
            arraySizeY = arraySize[1]
            
            totalValues = arraySizeX*arraySizeY
            
            #Setup the full data set based on the known information
            if jj == 0:
                allData = np.zeros((arraySizeX,arraySizeY,head2[3]),dtype=Type)

            dataValues = np.fromfile(f,dtype=Type,count=totalValues)
            
            #reshape the data into an image with the correct shape
            dataValues = dataValues.reshape((arraySizeX,arraySizeY))
            
            allData[:,:,jj] = dataValues
        
        allData = np.squeeze(allData)
        xAxis = np.arange(calibrationX[0],-calibrationX[0],calibrationX[1])
        yAxis = np.arange(calibrationY[0],-calibrationY[0],calibrationY[1])
        
        #Maybe in the future use a nested dictionary to separate image/spectra and scan data:
        #dataOut = {}; imageData = {'images':allData,'pixelSizeX':pixelSizeX};dataOut['imageDictionary'] = imageData;
        dataOut = {'imageData':allData,'xAxis':xAxis,'yAxis':yAxis,'pixelSizeX':calibrationX[1],'pixelSizeY':calibrationY[1],'filename':fname,'scanCalibration':dimCalibration,'scanDescription':dimensionDescription,'scanUnit':scanUnit}
        
        return dataOut
            
    f.close()
#END serReader
