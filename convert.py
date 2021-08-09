
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
parser.add_argument("path", type=pathlib.Path)
parser.add_argument("--output_directory", type=pathlib.Path)

arguments = parser.parse_args()

file_path = arguments.path
if arguments.output_directory is not None:
    output_directory = arguments.output_directory

if not file_path.exists():
    raise Exception("Input file does not exist")
if not file_path.is_file():
    raise Exception("Input file path is not a file")
if file_path.suffix.lower() == '.txt':
    #directory = 'E:\\16-07-21_Temp\\15-07-2021 testing\\deparafinized-large-area'
    #file_format = 'mb_x-1V,y-1.1Vstep0.005_'
    
    directory = file_path.parents[0]
    file_name = file_path.stem # Filename without the extension.

    match = re.match(r'^(.*)[b]\d+$', str(file_name)) # It's just a name ending in a b + a number.
    if match != None:
        file_format = match.group(1) # E.g. "mb_x-1V,y-1.1Vstep0.005_"
        
        full_array = read_txt_array_scan(directory, file_format)
        
        print_memory_usage()

        # Build the Intensity array.
        print('Building Intensity Array')
        intensity_array = build_intensity_array(full_array)
        
        # The txt file machine seems to generate a bunch of noise (high levels of intensity)
        # at the top of the sample. Should strip this. Probably just run down.
        intensity_array = remove_top_noise(intensity_array)

        # Could cache the intensity matrix.
        #np.save(directory + file_format.format('') + '.intensity.npy', intensity_array)
    else:
        raise Exception('Unexpected txt file format')
elif file_path.suffix.lower() == '.npy':
    # E.g. "C:\\Users\\swes043\\Honours\\OCT\\1300_SS\\npy files_16th May_1300nm_SS\\grid01_Int.npy"
    
    intensity_array = np.load(file_path)
else:
    raise Exception('Unexpected input file path')


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
    print('Using threshold:', threshold)
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

title = 'Path: ' + str(file_path) + '\n'
title += 'Heatmap Alg: ' + str(heatmap_algorithm)
title += ',Intensity Dim: ' + str(intensity_array.shape)
if voxel_dimensions:
    title += ',Voxel Dim: ' + str(voxel_dimensions)


attenuation_viewer.view_attenuation(title, intensity_array, heatmap_array, projection_array)

#print("Generating attenuation")
#generate_attenuation.generate_attenuation(intensity_array, intensity_array, output_directory)

#print("Generating heatmap")
#generate_heat_map.generate_heat_map(intensity_array, intensity_array, output_directory)

        






    
    
    
    
    
    
    
    
    