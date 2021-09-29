
# Need to take 850_SD *.txt files and convert them into a NPY?? format.
# C:\\Users\\swes043\\Honours\\OCT\850_SD\\brain_1stJune_850nm_SD
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
import psutil
import pathlib
import re
import math
from nptdms import TdmsFile
import tqdm # Progress bar


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

    # Get the dimensions from the first one.
    arr = read_txt_array_file(format_txt_path(directory, file_format, 0))

    return [b, len(arr), len(arr[0])]

def read_txt_array_scan_single(directory, file_format, dimensions, raw_array, raw_array_lock, b):
    file_path = format_txt_path(directory, file_format, b)

    print('reading file: ' + str(b + 1) + '/' + str(dimensions[0]))

    arr = read_txt_array_file(file_path)

    if raw_array_lock is None:
        raw_array[b] = arr
    else:
        raw_array_lock.acquire()
        try:
            raw_array[b] = arr
        finally:
            raw_array_lock.release()


# Expects a directory (e.g. C:\\User\\swes043\\OCT) and 
# file name format (e.g. "mb_x-1V,y-1.1Vstep0.005_" for files like mb_x-1V,y-1.1Vstep0.005_b0.txt)
def read_txt_array_scan(directory, file_format):
    multithreaded = False # Seems to be CPU bound, so this isn't actually improving things.

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

            pool = multiprocessing.pool.ThreadPool(10)
            for b in range(0, dimensions[0]):
                pool.apply_async(read_txt_array_scan_single, (directory, file_format, dimensions, raw_array, raw_array_lock, b,))
            pool.close()
            pool.join()
        else:
            for b in range(0, dimensions[0]):
                read_txt_array_scan_single(directory, file_format, dimensions, raw_array, None, b)

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
            intensity_array = build_intensity_array(raw_array, False)
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

    # Taken from tdmsCode.py. I guess you could check if this is divisible, but w/e.
    a_scan_length = int(len(data) / b_scan_num / a_scan_num)

    data.resize((b_scan_num, a_scan_num, a_scan_length))
    raw_array = np.array(data)

    return raw_array

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print('Memory Usage (MB): ' + str(process.memory_info().rss / 1024 ** 2))

