
# Need to take 850_SD *.txt files and convert them into a NPY?? format.
# C:\Users\swes043\Honours\OCT\850_SD\brain_1stJune_850nm_SD
# File names
# - brain_x-1V,y-1.1Vstep0.0025_b0.txt (b0 -> b226)
# - brain_x-2V,y-2Vstep0.0025_b0.txt (b0 -> b1600)
# - sample2_brain_x-2V,y-2Vstep0.0025_b0.txt (b0 -> b1600)

# I think each of these represents a B scan (one for each movement of the laser). (e.g. sample 2 moves 1600 times)

# Firstly, what is the format of these .txt files?
# - Each is essentially a 2 dimensional array of integers.
# - This represents a single channel (Intensity (?))
# - 1V example - each file has 201 rows (depth (?)), and 2048 columns.


# Apparently need to take FFT, then the absolute value. (See Scan_processing_2d, file = tdmsCode).

# Should have a look at raw NPY data (before Scan_processing_Xd)


# Dimensions:
#  First dimension is "running into the screen". One per B scan.
#  Second dimension is the "depth", running into the sample from the top.
#  Third dimension is "width", running along the sample. Like A scans I guess.

import os
import numpy as np
import numpy.ma as ma
import psutil
import pathlib
import re
import math
from nptdms import TdmsFile

# Library from physics department containing TDMS code.
import reference_code.tdmsCode as pytdms

# Read txt array file into numpy array.
def read_txt_array_file(file_path):
    with open(file_path) as f:
        content = f.readlines()
    
    # Construct the 2 dimensional array
    arr = np.array([ [int(element) for element in line.split()] for line in content ])
    
    return arr

def format_txt_path(directory, file_format, b_num):
    return pathlib.Path.joinpath(directory, file_format + 'b' + str(b_num) + '.txt')
    
# Returns array containing dimensions for this scan.
def scan_txt_dimensions(directory, file_format):
    b = 0
    
    while True:
        file_path = format_txt_path(directory, file_format, b)
        if not file_path.is_file():
            # Reached the last array file.
            break
        
        b += 1
    
    arr = read_txt_array_file(format_txt_path(directory, file_format, 0)) # Get the dimensions from the first one.
    return [b, len(arr), len(arr[0])]
    

#raw_array_lock = multiprocessing.Lock()
#threaded_raw_array = None
    
def read_txt_array_scan_single(directory, file_format, dimensions, raw_array, raw_array_lock, b):
    file_path = format_txt_path(directory, file_format, b)
    
    print('reading file: ' + str(b + 1) + '/' + str(dimensions[0]))
    
    arr = read_txt_array_file(file_path)
    
    raw_array_lock.acquire()
    try:
        raw_array[b] = arr
    finally:
        raw_array_lock.release()
    


# Expects a directory (e.g. C:\User\swes043\OCT) and 
# file name format (e.g. "mb_x-1V,y-1.1Vstep0.005_" for files like mb_x-1V,y-1.1Vstep0.005_b0.txt)
def read_txt_array_scan(directory, file_format):
    multithreaded = True

    cache_file_path = pathlib.Path.joinpath(directory, file_format + 'b.txt.cache.npy')

    # Check if cache file exists
    if cache_file_path.is_file():
        print('Reading cache file ({})'.format(cache_file_path))
        raw_array = np.load(cache_file_path)
    else:
        dimensions = scan_txt_dimensions(directory, file_format)

        print('columns = {}'.format(str(dimensions[2])))
        print('rows = {}'.format(str(dimensions[1])))    # Number of lines in a txt file.
        print('depth = {}'.format(str(dimensions[0])))   # Number of txt files.

        raw_array = np.empty(dimensions)

        if multithreaded:        
            import multiprocessing
            import multiprocessing.pool
            
            raw_array_lock = multiprocessing.Lock()
            
            pool = multiprocessing.pool.ThreadPool(50)
            for b in range(0, dimensions[0]):
                pool.apply_async(read_txt_array_scan_single, (directory, file_format, dimensions, raw_array, raw_array_lock, b,))
            pool.close()
            pool.join()
        else:
            for b in range(0, dimensions[0]):
                read_txt_array_scan_single(directory, file_format, dimensions, raw_array, raw_array_lock, b)
        
        #thread_pool = multiprocessing.Pool(processes = 20)
        #result = thread_pool.starmap(read_txt_array_scan_single, 
        #    [ (directory, file_format, dimensions, raw_array, raw_array_lock, b) for b in range(0, dimensions[0]) ])
            
        np.save(cache_file_path, raw_array)
    
    return raw_array
    
def load_txt_intensity_array(file_path):
    #directory = 'E:\\16-07-21_Temp\\15-07-2021 testing\\deparafinized-large-area'
    #file_format = 'mb_x-1V,y-1.1Vstep0.005_'
    
    directory = file_path.parents[0]
    file_name = file_path.stem # Filename without the extension.

    match = re.match(r'^(.*)[b]\d+$', str(file_name)) # It's just a name ending in a b + a number.
    if match != None:
        file_format = match.group(1) # E.g. "mb_x-1V,y-1.1Vstep0.005_"
        
        intensity_cache_file_path = pathlib.Path.joinpath(directory, file_format.format('') + '.intensity.cache.npy')
        if intensity_cache_file_path.is_file():
            # Just read the intensity cache file if it is there.
            print('Reading intensity cache file', intensity_cache_file_path)
            intensity_array = np.load(intensity_cache_file_path)
        else:
            # Have to read the txt files if no intensity cache file.
            raw_array = read_txt_array_scan(directory, file_format)
        
            print_memory_usage()          
        
            # Build the Intensity array.
            print('Building Intensity Array')
            intensity_array = build_intensity_array(raw_array)
            np.save(intensity_cache_file_path, intensity_array)
        
        return intensity_array
    else:
        raise Exception('Unexpected txt file format')

# Expects number of A scans and B scans that are present.
def read_tdms_array(file_path, a_scan_num, b_scan_num):
    
    tdms_file = TdmsFile(file_path)
    
    # tdms_file.groups() - returns all groups. group = tdms_file['Name']
    # group.channels() - returns all channels. channel = group['Name']
    data = tdms_file['Untitled']['Ch1'].data
    
    # Taken from tdmsCode.py
    a_scan_length = int(len(data) / b_scan_num / a_scan_num)
    
    data.resize((b_scan_num, a_scan_num, a_scan_length))
    raw_array = np.array(data)

    return raw_array
        
def build_intensity_array(raw_array):
    
    # Resultant array is cut in half (in third dimension) and rotated.
    intensity_array = np.empty((raw_array.shape[0], int(raw_array.shape[2] / 2), raw_array.shape[1]))

    for i in range(0, raw_array.shape[0]):
        b_scan = raw_array[i]
        
        # Absolute(Fourier Transform( B Scan ) ) 
        arr = np.fft.fft(b_scan)
        arr = np.absolute(arr)
        
        # Log10 (?). Would need to adjust the colour map when visualising
        #arr = np.log10(arr) # np.log(arr);
        
        #rot = np.rot90(np.sqrt(abs_ch0 ** 2 + abs_ch1 ** 2)[:, 0:int(A_length / 2)], 3) # Including Retardation.
        
        arr = np.rot90(arr[:, 0:int(len(arr[0]) / 2)], k = 3) # Only need half of the resultant array. Rotated by 270 degrees (k=3(?)).
        # Should really flip the image too (so as it appears in the same orientation as the labview software).
        arr = np.flip(arr, 1)
        
        intensity_array[i] = arr
        
    return intensity_array
        
def print_memory_usage():
    process = psutil.Process(os.getpid())
    print('Memory Usage (MB): ' + str(process.memory_info().rss / 1024 ** 2))
    
def build_intensity_mean_array(intensity_array):
    intensity_mean_array = []
    vox_c = 5 # Voxel dimension (?)

    # Mean of x (and next 4) in first dimension. Size of a voxel?
    # Essentially seems to just create a smaller array (smaller first dimension, compressed). Why though?
    for x in range(0, intensity_array.shape[0], vox_c):
        mean = np.mean((intensity_array[x:x + vox_c, :, :]), axis = 0) 
        intensity_mean_array.append(mean)
        
    return np.array(intensity_mean_array)

# Voxel mean array, allows adjustment in all 3 dimensions.
# TODO: Re-test this
def build_intensity_mean_array_2(intensity_array, voxel_dimensions):
    shape = intensity_array.shape
    voxel_dimensions = np.array(voxel_dimensions)
    
    voxel_array = np.empty(shape // voxel_dimensions)

    for i in range(0, shape[0] // voxel_dimensions[0]):
        for j in range(0, shape[1] // voxel_dimensions[1]):
            for k in range(0, shape[2] // voxel_dimensions[2]):
                offset_0 = i * voxel_dimensions[0]
                offset_1 = j * voxel_dimensions[1]
                offset_2 = k * voxel_dimensions[2]
                
                voxel_array[i, j, k] = np.mean(intensity_array[
                    offset_0 : offset_0 + voxel_dimensions[0],
                    offset_1 : offset_1 + voxel_dimensions[1],
                    offset_2 : offset_2 + voxel_dimensions[2]])
                    
    return voxel_array
    
    
def build_attenuation_map(intensity_array, voxel_dimensions):
    # Input should already have the surface rolled.
    # Input should include the dimensions of the voxels.
    # Down each A scan (?? length) we need to calculate the slope of the log of the intensities.
    #    Then within each voxel, take the average of these slopes.
    # Dimension 3, should be down the A scan
    
    shape = intensity_array.shape
    voxel_dimensions = np.array(voxel_dimensions)
    
    voxel_array = np.empty(shape // voxel_dimensions)
    
    depth_range = np.arange(0, voxel_dimensions[1])
    
    for i in range(0, shape[0] // voxel_dimensions[0]):
        for j in range(0, shape[1] // voxel_dimensions[1]):
            for k in range(0, shape[2] // voxel_dimensions[2]):
                offset_0 = i * voxel_dimensions[0]
                offset_1 = j * voxel_dimensions[1]
                offset_2 = k * voxel_dimensions[2]
            
                # Run down the depth and take the mean at each "layer".
                # Depth is within the second dimension (dim[1])
                mean_array = []
                for m in range(0, voxel_dimensions[1]):
                    mean_array.append(np.mean(intensity_array[
                        offset_0 : offset_0 + voxel_dimensions[0],
                        offset_1 + m,
                        offset_2 : offset_2 + voxel_dimensions[2]]))
                    
                with np.errstate(divide = 'ignore'): # Ignore divide by zeros here.
                    log_vals = np.log(mean_array)
                
                # Find the slope of these log(means) and this is the attenuation in this voxel.
                # TODO: Change to numpy.polynomial
                fit = np.polyfit(depth_range, log_vals, 1)
                slope = fit[0]
                
                voxel_array[i, j, k] = slope
                
    return voxel_array
    
    
def build_heatmap_array(intensity_array):
    ### define voxel size
    voxC = 3  # number of B-scans to average
    voxA = 20 #20# int(40*voxC/10) # size of A-scan 
    voxB = 20 #20# int(40*voxC/14) # size of B_scan segments

    # This calculates the attenuation for each voxel. Is at least O(n^3), something like O(n^5) actually, I think,
    #     so can be slow. Maybe could possibly multithread this (?). Very slow actually.
    # TODO: Need to have a look at this code and check see it works, seemed quite complicated.
    #     : One would think that it needs to examine the attenuation drop off (the slope)
    #     : Should really just be peak differential I would think (the steepest point), based
    #     : on how it has been described to me before.
    atten_c, mask, _ = pytdms.Heatmap_Int(intensity_array, voxC = voxC, voxB = voxB, voxA = voxA)
    
    # Not sure what the point in this is.
    masked_c = ma.array(atten_c, mask = mask)
    masked_c = ma.compressed(masked_c) # Apparently returns all non-masked data as 1d array. 
    masked_c[np.isnan(masked_c)] = 0
    
    return atten_c
    
# Just find the maximum heatmap value down each projection. O(n^3), unavoidable I guess.
def build_heatmap_max_projection_array(heatmap_array):
    result = np.empty((heatmap_array.shape[0], heatmap_array.shape[2]))
    
    # Probably a better numpy way of writing this I would think.
    for i in range(0, heatmap_array.shape[0]):
        for j in range(0, heatmap_array.shape[2]):
            max_atten = math.inf
            
            # FIXME: Run from 5% in, to 2/3 down, bit hacky, could work out where the surface was rolled.
            # Maybe the depth thing in the example code stores this information. That would make sense.
            for k in range(int(heatmap_array.shape[1] * 0.05), int(heatmap_array.shape[1] * (2.0 / 3.0))):
                max_atten = min(max_atten, heatmap_array[i, k, j])
            result[i, j] = max_atten
            
    return result
    
# TODO: Idea is to trace down from the top, find the maximum slope (differential).
# Surely this is the best way of finding an attenuation drop off.
def build_max_differential_projection_array(intensity_array):
    pass

    
# TODO: Calculate using the formula Abi used. See TDMSProcessBrainAbi.py - 
# I don't really see what this is doing though, the resultant image is weird.
def build_projection_array(intensity_array):
    pass

# Remove the noise at the top of the intensity array.
def remove_top_noise(intensity_array):
    # Best method I think would be to find the depth that contains values > mean. And strip these.
    return intensity_array[:, 20:, :]
    
# Surface detection, Rolls stuff at the top to the bottom.
# In place, will replace intensity_array values.
def surface_roll(intensity_array, threshold):
    z = 0
    for x in intensity_array:
        # This function defines surface as the point where "length" number of values (starting at index 0)
        #     have exceeded the threshold.
        # Strange really, would prefer something else (surface is the first dark/high intensity area, 
        # could do something with that).
        surface = np.array(pytdms.surface_detect(x, threshold = threshold, length = 5, skip = 5))

        for y in range(len(surface)):
            x[0:surface[y], y] = 0
            x[:, y] = np.roll(x[:, y], -1 * surface[y])

        z += 1
        
# Find surface and depth. Depth will be useful when generating attenuation heatmap.
def find_surface_and_depth(intensity_array, threshold):

    # Ideally would like to use the neighbouring A scans to improve accuracy of surface.
    # Could even look in 2 dimensions.
    # Really, The surface is the area where the intensity increases rapidly.
    # Could use linear regression (?). Maybe is also what we should be using in the voxel 
    # attenuation calculation, may be a lot faster than that polynomial calculation
    pass
        
def power_law_transform(intensity_array):
    transform_array = np.empty(intensity_array.shape)

    # From Abi's example. Taken from https://code.tutsplus.com/articles/image-enhancement-in-python--cms-29289
    # Also known as gamma correction.
    # p(i,j) = kI(i,j)^gamma. k and gamma are constants. k = 1, gamma = 1.5 here I believe.
    # This doesn't make a lot of sense. Gamma correction is an operation applied to RGB pixels 
    # and can help due to the way computer displays display pixels.
    
    for i, b_scan in enumerate(intensity_array):
        transform = b_scan
        # Why would you do this? Dividing by 255 is to normalise the pixel value (RGB)
        # https://stackoverflow.com/questions/20486700/why-do-we-always-divide-rgb-values-by-255#:~:text=Since%20255%20is%20the%20maximum,255%20since%200%20is%20included.
        
        # https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
        # You see, when twice the number of photons hit the sensor of a digital camera, 
        # it receives twice the signal (a linear relationship). However, that’s not how 
        # our human eyes work. Instead, we perceive double the amount of light as only 
        # a fraction brighter (a non-linear relationship)! Furthermore, our eyes are also 
        # much more sensitive to changes in dark tones than brighter tones (another 
        # non-linear relationship).
        # In order to account for this we can apply gamma correction, a translation 
        # between the sensitivity of our eyes and sensors of a camera.
        
        transform = transform / 255.0  
        
        transform = transform ** 1.5   # cv2.pow(adjusted, 1.5) # ????? Why not b_scan ** 1.5?
        
        transform_array[i] = transform
    
    return transform_array
    
    
    