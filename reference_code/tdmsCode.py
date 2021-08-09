from nptdms import TdmsFile,TdmsWriter,ChannelObject
import matplotlib.pylab as plt
import numpy as np
from PIL import Image
from scipy import signal,ndimage
import numpy.ma as ma
import os as os
import glob as glob
import struct

try:
    from C_tdms import depth_detect, surface_detect, atten_Int, Scan_processing_3D, Scan_processing_2D
except ImportError:
    print('Could not load C_tdms, falling back to python')
    def depth_detect(Mat, surface, threshold, length = 10):
    ## This function operates similar to surface_detect. If all the next n = length values are below the threshold then record the surface location
        depth = np.zeros_like(surface, dtype =int)
        for i in range(0,len(Mat[0,:])):
            count = 0
            for j in range(surface[i],len(Mat[:,i]-length)):
                if Mat[j,i] < threshold:
                    count += 1
                else:
                    count = 0
		
                if count == length:
                    depth[i] = j - length
                    break
            else:
                depth[i] = len(Mat[:,i] - length)

        return(depth)


    def surface_detect(Mat, threshold, length = 10, skip = 0):

    # This function detects the surface location of a sample in an OCT image. Inputs are the B-scan Matrix (Mat), threshold is the value used to find the surface and length is used to minimize triggering on false surfaces.
        surface = np.zeros_like(Mat[0,:], dtype = int)
        for i in range(0, len(Mat[0,:])): #Index through columns
            count = 0
            for j in range(0,len(Mat[:,i])- length): # Index through intensity values
                if Mat[j,i] > threshold:
                    count += 1
                else:
                    count = 0
                if count == length:
                    surface[i] = j + skip
                    break
            else:
                surface[i] = len(Mat[:,i])- length
        return(surface)


    def Scan_processing_3D( C_scan_CH0, C_scan_CH1, string, name, save = False):
    #### Turn the channel data into Intensity and Birefringence matrices

    ## Inputs 3D numpy matrices for each channel. Outputs are the Intenstiy and Retardation  3D matrices. If save == True, the processed data is saved as a numpy array using the provided name.
        Intensity = []
        Retardation = []
    ### for each B_scan 
        for x in range(0,len(C_scan_CH1[:,0,0])):

        ### for each channel compute the Intensity (Int) and Retardation (Ret).
            ch1 = C_scan_CH1[x,:,:]
            ch0 = C_scan_CH0[x,:,:]
            Int, Ret = Scan_processing_2D(ch0,ch1)

        ### append the B_scan to the C_scan
            Intensity.append(Int)
            Retardation.append(Ret)
    #### save the data to a .npy file    
        if save == True:    
            np.save(string + name + 'Int.npy', np.array(Intensity))
            np.save(string + name + 'Ret.npy', np.array(Retardation))
        return np.array(Intensity), np.array(Retardation)

    def Scan_processing_2D(B_scan_CH0, B_scan_CH1):
    
    ## Inputs 2D numpy matrices for a B_scan in each channel. Outputs are the Intenstiy and Retardation  2D matrices.
    
        for x in range(len(B_scan_CH0)):
            B_scan_CH1[x] = np.hanning(len(B_scan_CH1[x]))*B_scan_CH1[x]
            B_scan_CH0[x] = np.hanning(len(B_scan_CH0[x]))*B_scan_CH0[x]

        A_length = len(B_scan_CH1[0])
        fft_ch0 = np.fft.fft(B_scan_CH0)
        fft_ch1 = np.fft.fft(B_scan_CH1)
    
    #Take the asbolute
        abs_ch0 = np.absolute(fft_ch0)
        abs_ch1 = np.absolute(fft_ch1)

    #Now create the image from the data
        Int = np.rot90(np.sqrt(abs_ch0**2 + abs_ch1**2)[:,0:int(A_length/2)],3)
        Ret = np.rot90(np.arctan(abs_ch1/abs_ch0)[:,0:int(A_length/2)],3)
        Int = Int[:,0:(np.size(B_scan_CH0,0))]
        Ret = Ret[:,0:(np.size(B_scan_CH0,0))]
        return(Int, Ret)

    def atten_Int(Intmean, Depth,voxA, voxB, jumpA):
        len_a = range(0,len(Intmean[0,:,0])- voxA, jumpA)
        len_b = range(0,len(Intmean[0,0,:]), voxB)
        masked_c = np.zeros((np.shape(Intmean)[0],len(len_a),len(len_b)))
        atten_c = np.zeros_like(masked_c)
        for x in range(0,np.shape(Intmean)[0]):
            for k,z in enumerate(len_b):
                end = np.mean(Depth[x,z:z+voxB]).astype(int)
                vox = np.mean(Intmean[x,:,z:z+voxB], axis = 1)
                with np.errstate(divide = 'ignore'): # Ignore divide by zeros here.
                    logmeanA = np.log(vox)
                logmeanA[np.argwhere(np.isnan(logmeanA))] = logmeanA[np.argwhere(np.isnan(logmeanA)) - 1]

                for j,y in enumerate(len_a):
                    #print(y,j)
                    p = np.polyfit(np.arange(0,len(logmeanA[y:y+voxA])), logmeanA[y:y+voxA], 1)
                    atten_c[x,j,k] = p[0]
                    if y+voxA < end:
                        pass
                    else:
                        masked_c[x,j,k] = 1
                #atten_b.append(atten_a)
                #masked_b.append(masked_a)
            #atten_c.append(np.flip(np.rot90(atten_b,3),1))
            #masked_c.append(np.flip(np.rot90(masked_b,3),1))
            #atten_c = (atten_c)
        return(atten_c, masked_c)

    
LAMBDA_CONSTANT= 1310*10**-9 ## Wavelength of OCT system
NREF = 1.4 ## Refractive index of sample

### This module contains a collections of functions to read and process PS_OCT data

### Example of use:

### Read the raw data and convert to C_Scan matrix using the read_tdms function. The number of B_scans and A_scans are required to shape the C_scan:

#A_scan_num = 714
#B_scan_num = 150

#Data0  =  pytdms.read_tdms("Raw_C-Scan/G0_S1_XY_Ch0.tdms", A_scan_num, B_scan_num)
#Data1  =  pytdms.read_tdms("Raw_C-Scan/G0_S1_XY_Ch1.tdms", A_scan_num, B_scan_num)

###  Process the raw PS_OCT data to find the intensity and retardation matrices using the Scan_processing_3D function. If save == True the processed data will be saved as a numpy arrays with the given name.

#Int, Ret = pytdms.Scan_processing_3D(Data0,Data1,'Raw_C-Scan/G0_S1_XY',save = True)

### If you just want to process a single B_Scan you can use the Scan_processing_2D function. 

#Int_2D, Ret_2D = pytdms.Scan_processing_2D(Data0[50,:,:],Data1[50,:,:])

### to save the C_scan or B_scan as a tiff you can use the Save_Image funtion:

## to save a C_Scan:
#pytdms.Save_Image('Raw_C-Scan/G0_S1_XY_Int.tiff', Int)

## to save a B_Scan:
#pytdms.Save_Image('Raw_C-Scan/G0_S1_XY_Int.tiff', Int[50])

def read_frg(filename, B_scan_skip):

    with open(filename, 'rb') as binary_file:
        header = binary_file.read(512)
        width = int.from_bytes(header[20:24], byteorder = 'little')
        depth = int.from_bytes(header[24:28], byteorder = 'little')
        #print(width, depth)
        B_scan_num = int.from_bytes(header[28:32], byteorder = 'little')
        #C_scan_num = int.from_bytes(header[32:36], byteorder = 'little')
        #print(C_scan_num,B_scan_num)
        FFT_length = int.from_bytes(header[36:40], byteorder = 'little')
        #Record_length = int.from_bytes(header[44:48], byteorder = 'little')
        #print(FFT_length, Record_length)
        #PSOCT = int.from_bytes(header[48:50],byteorder = 'little')

        #Frame_size_bytes = int.from_bytes(header[40:44], byteorder = 'little')
        #Frame_size = width*FFT_length*2*2
        matrix = np.zeros((B_scan_num, depth, width))
        #print(np.shape(matrix))
        
        for x in range(1,2):

            Frame = binary_file.read(39)#Frame_size_bytes)
            header2 = Frame[0:40]
            #print(Frame[40:42])
            a_scan = []
            a_scan2 = []
            count = 0
            for x in range(0,int(B_scan_num)):
                for y in range(0,FFT_length,2):    
                    a_scan.append(int.from_bytes(binary_file.read(2), byteorder = 'little', signed = True))
                for z in range(0,FFT_length,2):
                    a_scan2.append(int.from_bytes(binary_file.read(2), byteorder = 'little', signed = True))
            a_scan = np.array(a_scan) 
            b_scan = np.reshape(a_scan, (FFT_length,-1))
            a_scan2 = np.array(a_scan2) 
            b_scan2 = np.reshape(a_scan2, (FFT_length,-1))
            print(np.shape(b_scan))

            Channel_A = np.fft.fftshift(np.fft.fft(b_scan[0:len(b_scan[:,0]):2,:] ))
            Channel_B = np.fft.fftshift(np.fft.fft(b_scan[0:len(b_scan2[:,0]):2,:] ))
            plt.plot(b_scan[0])
            plt.show()
            plt.imshow(abs(Channel_A**2 + Channel_B**2), cmap = 'binary', vmax = 10**12, vmin = 10**10)
            plt.show()
            
        binary_file.close()
    return(matrix)



def read_tdms(filename, A_scan_num, B_scan_num):

    ## Read the TDMS data from the PS-OCT system and write to a C-Scan matrix.

    ## inputs filename = TDMS file location, A_scan_num is the number of A_Scans in each B_Scan and B_scan_num is the number of B_Scans in the C_Scan.
    
    tdms_file = TdmsFile(filename) ##import the data as a TDMS
    data =  tdms_file.object('Untitled', tdms_file.group_channels('Untitled')[0].channel).data ## Extract the data
    
    A_scan_length = int(len(data)/B_scan_num/A_scan_num) # calculate A_scan length
    data.resize((B_scan_num,A_scan_num,A_scan_length))
    C_scan = np.array(data) 
    del data    
    return(C_scan)


def Save_Image(ImName, Mat):

    Mat = (np.amax(Mat)/Mat*255).astype(np.int16)
    # This function saves the convert matrix to a greyscale image    

    if Mat.ndim > 2:
        # make a list of images from Mat
        imlist = []
        for m in Mat:
            imlist.append(Image.fromarray(m))
        #save multiframe image
        imlist[0].save(ImName, save_all=True, append_images=imlist[1:])

    else:    
        im = Image.fromarray(Mat)
        im.save(ImName)

##### Below are functions for analysis of the OCT images



def birefringence(Ret, start, end, wavelength = LAMBDA_CONSTANT, nref = NREF):
    #Calculate the birefringence of an A-Scan 

    ### Select part of the image for analysis
    RetSection = Ret[start:end]

    ### Smooth the image a bit
    smoothRet = ndimage.filters.gaussian_filter(RetSection,3,0)

    ### calculate the depth of the image
    depthpix = (end-start)*(10/nref)*(10**(-6))

    ### calculate the cumulative change in phase
    CumulativePhase = np.cumsum(np.absolute(np.diff(smoothRet)))*wavelength/(depthpix)
    return(CumulativePhase)


def AttCoeff(A_scan,start,end):
    p = np.polyfit(np.arange(start,end,1), A_scan[start:end], 1)
    return(p)    

def Voxel_Ret(vox):
    meanA = np.mean(np.mean(vox, axis = 0), axis = 1)
    phase = birefringence(meanA,0,len(meanA))
    biref = phase[-1]        
    return(biref)

def Heatmap_Int(Int, voxC=5, voxB=15, voxA = 20):
    jumpA = 5
    jumpB = 2
    jumpC = 2
    #set up arrays
    atten_b =[]
    atten_a = []
    atten_c = []
    Intmean = []
    Depth = []
  # first average the B-scans 
    for x in range(0,np.shape(Int)[0], voxC):
        mean = np.mean((Int[x:x+voxC,:,:]),axis = 0)
        threshold = np.mean(mean)
        surface = np.zeros(len(mean[0,:])).astype(int)
        depth1 = depth_detect(mean,surface,threshold/2, length=5)
        depth = signal.savgol_filter(depth1, 51,2).astype(int)
        #plt.figure()
        #plt.imshow(mean,cmap = 'binary')
        #plt.plot(surface, 'r')
        #plt.plot(depth,'g')
        #plt.clim(6000,20000)
        #plt.colorbar()
        #plt.show()
        Depth.append(depth)
        Intmean.append(mean)

    Intmean = np.array(Intmean)
    Depth = np.array(Depth)
    # for each averaged B-scan divide the image into voxels and calculate the attenuation

    atten_c, masked_c = atten_Int(Intmean, Depth, voxA, voxB,jumpA)
    return(atten_c, masked_c, Intmean)

def Roll_image(Im, threshold, length = 5, skip = 10):
    Depth = []
    Surface = []
    im_zeros = np.zeros_like(Im)
    for i,x in enumerate(Im):
        surface1 =  surface_detect(x, threshold,length = 5, skip = 10)
        surface = signal.savgol_filter(surface1, 51,2).astype(int)
        Surface.append(surface)
        #depth1 = depth_detect(x,surface,threshold/3, length=5)
        #depth = signal.savgol_filter(depth1, 51,2).astype(int)
        #Depth.append(depth)
       # print(len(surface))
        #print(np.shape(x))
        for y in range(len(surface)):
            #print(surface[y])
            data = x[surface[y]:,y]
            #print(np.shape(data))
            im_zeros[i,0:len(data),y] =  data
    Im[:,:,:] = im_zeros
    return (np.array(Surface))#, np.array(Depth))
    

def hist_output(save_location, data_location, B_averages, show = True):
    if not os.path.exists(save_location):
                os.makedirs(save_location)
    for file in glob.glob(data_location + "grid*_Int.npy"):
        gridname = os.path.splitext(file)[0].replace(data_location, '')
        print (save_location + gridname)
        if not os.path.isfile(save_location + gridname + '_masked.npy'):

            Int = np.load(file)
            #Int = np.array(Int[0:150,0:500,0:700])
            Int = np.array(Int[0:150,0:500,90:660])


            Surface = Roll_image(Int,threshold = np.mean(Int))

        ### define voxel size
            voxC = B_averages # number of B-scans to average
            #voxA =int(40*voxC/10) # size of A-scan 
            #voxB =int(40*voxC/14) # size of B_scan segments
            #voxA = 30
            #VoxB = 40
        ##calculate the attenuation for each voxel
            atten_c, mask, Intmean = Heatmap_Int(Int,voxC = voxC, voxB = 30, voxA = 40)
            masked_c = ma.array(atten_c, mask = mask)
        #print(masked_c)
            masked_c = ma.compressed(masked_c)
            print(len(masked_c))
            masked_c[np.isnan(masked_c)] = 0
            np.save(save_location + gridname + '_masked.npy', masked_c)
        #hist, edges = np.histogram(masked_c, bins = 100, density=False)
        #np.save(save_location + gridname + '_hist.npy', np.array([hist, edges]))
            if show == True:
                plt.figure()
                plt.hist(masked_c, bins = 100)
                plt.xlabel('Attenuation[cm-1]')
                plt.ylabel('Number of voxels')
                plt.show()
        

def FFT_output(save_location, data_location, B_averages, show = True):
    if not os.path.exists(save_location):
                os.makedirs(save_location)
    for file in glob.glob(data_location + "grid*_Ret.npy"):
        gridname = os.path.splitext(file)[0].replace(data_location, '')
        print (save_location + gridname)
        if not os.path.isfile(save_location + gridname + '_FFT.npy'):

            Ret = np.load(file)
            Ret = np.array(Ret[0:150,0:500,0:700])
            FFT = []
        ### define voxel size
            voxC = B_averages # number of B-scans to average
            for x in range(0,np.shape(Ret)[0], voxC):
                mean = np.mean((Ret[x:x+voxC,:,:]),axis = 0)
                FFT.append(np.fft.fftshiftnp.fft.fft(mean, axis = 1))
                print(np.shape(FFT))
                #calculate the attenuation for each voxel

            np.save(save_location + gridname + '_FFT.npy', )
        #hist, edges = np.histogram(masked_c, bins = 100, density=False)
        #np.save(save_location + gridname + '_hist.npy', np.array([hist, edges]))
        
def slope_intercept (x_val,y_val):
    x=x_val
    y=y_val
    m = ((np.mean(x)*np.mean(y))-np.mean(x*y))/((np.mean(x)*np.mean(x))-np.mean(x*x))
    m=round(m,2)
    b=(np.mean(y)-np.mean(x)*m)
    b=round(b,2)
    
    return m,b
            