
# Input, Either 
# A TXT file, representing one of the B scans TXT files. OR, 
# An NPY file, representing the intensity matrix.
# TODO, allow to read TDMS file directly

import os
import numpy as np
import psutil
import argparse
import pathlib
import re
import math

import oct_library as library
#import generate_heat_map
#import generate_attenuation
import attenuation_viewer

# Library from physics department containing TDMS code.
import reference_code.tdmsCode as pytdms


def file_paths_display_string(file_paths):
    prefix = os.path.commonprefix(file_paths)
    
    return prefix + ' [' + ','.join([ str(file_path)[len(prefix) : ] for file_path in file_paths ]) + ']'
    
    

#print_memory_usage()

DEBUG = True
roll_surface = True
apply_power_law_transform = False
heatmap_algorithm = 2 # 1 = Abi version, 2 = My version

# Example code set maximum intensity as 20000. Not sure why. math.inf, no threshold.
maximum_intensity = 20000 # math.inf # 20000
#minimum_intensity = 1000

output_directory = "C:\\Users\\swes043\\Honours\\OCT_Data\\test_output"


parser = argparse.ArgumentParser()
parser.add_argument("path", type=pathlib.Path, action='append', nargs='+')
parser.add_argument("--output_directory", type=pathlib.Path)

arguments = parser.parse_args()

if arguments.output_directory is not None:
    output_directory = arguments.output_directory

intensity_array = None

file_paths = arguments.path[0]

for file_path in file_paths:
    print('Attempting to read: ', file_path)
    
    if not file_path.exists():
        raise Exception("Input file does not exist")
    if not file_path.is_file():
        raise Exception("Input file path is not a file")
    if file_path.suffix.lower() == '.txt':
        temp_intensity_array = library.load_txt_intensity_array(file_path)
        
        # The txt file machine seems to generate a bunch of noise (high levels of intensity)
        # at the top of the sample. Should strip this. Probably just run down.
        temp_intensity_array = library.remove_top_noise(temp_intensity_array)
        
    elif file_path.suffix.lower() == '.npy':
        # E.g. "C:\\Users\\swes043\\Honours\\OCT\\1300_SS\\npy files_16th May_1300nm_SS\\grid01_Int.npy"
        
        temp_intensity_array = np.load(file_path)
        
    elif file_path.suffix.lower() == '.tdms':
        
        a_scan_num = 714
        b_scan_num = 250
        
        # Expecting Ch1 (?)
        raw_array = library.read_tdms_array(file_path, a_scan_num, b_scan_num)
        
        temp_intensity_array = library.build_intensity_array(raw_array)
        
    else:
        raise Exception('Unexpected input file path')
        
    if intensity_array is None:
        intensity_array = temp_intensity_array
    else:
        # Should join in the first dimension. Will fail if the two don't have the same length in the other two dimensions.
        intensity_array = np.concatenate((intensity_array, temp_intensity_array), axis = 0)


#intensity_array[intensity_array < minimum_intensity] = 0
intensity_array[intensity_array > maximum_intensity] = maximum_intensity # Clamp to maximum
    
#if DEBUG:
    # Save the first intensity B scan.
    #np.savetxt(pathlib.Path.joinpath(directory, file_format + 'b0.pre.tmp.txt'), intensity_array[0], delimiter = '\t')
    
print("Intensity dimensions = " + str(intensity_array.shape))
#print(intensity_array)

intensity_mean_array = library.build_intensity_mean_array(intensity_array)
#print('Building mean map')
#intensity_mean_array_2 = build_intensity_mean_array_2(intensity_array, (10, 10, 10))

print('Intensity mean dimensions: ' + str(intensity_mean_array.shape))
#print('Intensity mean 2 dimensions: ', intensity_mean_array_2.shape)

if roll_surface:
    threshold = np.mean(intensity_array)
    print('Using surface threshold:', threshold)
    library.surface_roll(intensity_mean_array, threshold)
    library.surface_roll(intensity_array, threshold)
    #library.surface_roll(intensity_mean_array_2, threshold)
    
#if DEBUG:
    # Save the first intensity B scan.
    #np.savetxt(pathlib.Path.joinpath(directory, file_format + 'b0.post.tmp.txt'), intensity_array[0], delimiter = '\t')
    
if apply_power_law_transform:
    intensity_array = library.power_law_transform(intensity_array)
    
voxel_dimensions = None
print('Building attenuation map')
if heatmap_algorithm == 1:
    heatmap_array = library.build_heatmap_array(intensity_array)
    print('Heatmap dimensions: ' + str(heatmap_array.shape))
else: # 2
    voxel_dimensions = (10, 20, 10)
    heatmap_array = library.build_attenuation_map(intensity_array, voxel_dimensions)
    print('Attenuation map dimensions: ' + str(heatmap_array.shape))

print('Building projection array')
projection_array = library.build_heatmap_max_projection_array(heatmap_array)
#print(projection_array.shape)

title = 'Path: ' + file_paths_display_string(file_paths) + '\n'
title += 'Heatmap Alg: ' + str(heatmap_algorithm)
title += ',Intensity Dim: ' + str(intensity_array.shape)
if voxel_dimensions:
    title += ',Voxel Dim: ' + str(voxel_dimensions)


attenuation_viewer.view_attenuation(title, intensity_array, heatmap_array, projection_array)

#print("Generating attenuation")
#generate_attenuation.generate_attenuation(intensity_array, intensity_array, output_directory)

#print("Generating heatmap")
#generate_heat_map.generate_heat_map(intensity_array, intensity_array, output_directory)

        






    
    
    
    
    
    
    
    
    