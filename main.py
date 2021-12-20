
# Input, Either
# A TXT file, representing one of the B scans TXT files. OR,
# An NPY file, representing the intensity matrix. OR,
# A TDMS file, from labview. Reads raw 'Ch1' intensity data.

import pyximport; pyximport.install()

import os
import numpy as np
import psutil
import argparse
import pathlib
import re
import math
from time import perf_counter

import oct_library as library
import file_library as file_library
import attenuation_viewer

# Library from physics department containing TDMS code.
import reference_code.tdmsCode as pytdms

DEBUG = True

def file_paths_display_string(file_paths):
    prefix = os.path.commonprefix(file_paths)

    # TODO: Bug here if all paths are equal (displays [] at the end)
    return prefix + ' [' + ','.join([ str(file_path)[len(prefix) : ] for file_path in file_paths ]) + ']'

def main():
    roll_surface = True
    view_mean_array = False
    view_rolled_intensity = False
    apply_power_law_transform = False

    view_intensity_bounds = (0, 20000) # (Min, Max)
    heatmap_algorithm = 2 # 1 = Abi version, 2 = My version, 3 = Smoothed A scans.

    # Example code set maximum intensity as 20000. Not sure why. math.inf, no threshold.
    maximum_intensity = math.inf # 20000
    #minimum_intensity = 1000

    output_directory = '' # TODO: Not sure what this should be.
    #print_memory_usage()


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
            temp_intensity_array = file_library.load_txt_intensity_array(file_path)

            # Assuming .txt input is from the 800nm system.

            # The txt file machine seems to generate a bunch of noise (high levels
            # of intensity) at the top of the sample. Should strip this. Probably
            # just run down.
            temp_intensity_array = library.remove_top_noise(temp_intensity_array)

            # Bounds should be different for 800nm system.
            view_intensity_bounds = (0, 10000)

        elif file_path.suffix.lower() == '.npy':
            # E.g. "C:\\Users\\swes043\\Honours\\OCT\\1300_SS\\npy files_16th May_1300nm_SS\\grid01_Int.npy"

            temp_intensity_array = np.load(file_path)

        elif file_path.suffix.lower() == '.tdms':

            # 1300 output is a flat array it seems, so need to know the dimensions.
            a_scan_num = 714
            b_scan_num = 250

            # Expecting Ch1 (?)
            raw_array = file_library.read_tdms_array(file_path, a_scan_num, b_scan_num)

            temp_intensity_array = library.build_intensity_array(raw_array, True)

        else:
            raise Exception('Unexpected input file path')

        if intensity_array is None:
            intensity_array = temp_intensity_array
        else:
            # Should join in the first dimension. Will fail if the two don't have the same length in the other two dimensions.
            intensity_array = np.concatenate((intensity_array, temp_intensity_array), axis = 0)

    # TODO: Should maybe cache here, especially if there were multiple input files.

    #intensity_array[intensity_array < minimum_intensity] = 0
    intensity_array[intensity_array > maximum_intensity] = maximum_intensity # Clamp to maximum

    #if DEBUG:
        # Save the first intensity B scan.
        #np.savetxt(pathlib.Path.joinpath(directory, file_format + 'b0.pre.tmp.txt'), intensity_array[0], delimiter = '\t')

    print("Intensity dimensions = " + str(intensity_array.shape))
    #print(intensity_array)

    if view_mean_array:
        print('Building mean array')
        intensity_mean_array = library.build_intensity_mean_array(intensity_array)
        #intensity_mean_array_2 = library.build_intensity_mean_array_2(intensity_array, (5, 5, 5))

        print('Intensity mean dimensions: ' + str(intensity_mean_array.shape))
        #print('Intensity mean 2 dimensions: ', intensity_mean_array_2.shape)

    if roll_surface:
        threshold = np.mean(intensity_array)
        print('Using surface threshold:', threshold)
        print('Calculating surface positions')
        surface_positions, surface_positions_for_draw = library.find_surface(intensity_array, threshold)
        # Don't really like having two intensity_arrays sitting in memory, but w/e.
        rolled_intensity_array = library.build_rolled_intensity_array(intensity_array, surface_positions)

        #library.surface_roll(intensity_array, threshold)

        if view_mean_array:
            library.surface_roll(intensity_mean_array, threshold)
            #library.surface_roll(intensity_mean_array_2, threshold)

    #if DEBUG:
        # Save the first intensity B scan.
        #np.savetxt(pathlib.Path.joinpath(directory, file_format + 'b0.post.tmp.txt'), intensity_array[0], delimiter = '\t')

    if apply_power_law_transform:
        rolled_intensity_array = library.power_law_transform(rolled_intensity_array)

    voxel_dimensions = None
    heatmap_bounds = (-0.06, 0) # (Min, Max)
    print('Building attenuation map')
    time_start = perf_counter()
    if heatmap_algorithm == 1:
        heatmap_array = library.build_attenuation_map_1(rolled_intensity_array)
    elif heatmap_algorithm == 2:
        voxel_dimensions = (10, 20, 10)
        heatmap_array = library.build_attenuation_map_2(
            rolled_intensity_array, voxel_dimensions)
    elif heatmap_algorithm == 3:
        voxel_dimensions = (10, 1, 10)
        heatmap_array = library.build_attenuation_map_3(
            rolled_intensity_array, voxel_dimensions)
    else:
        raise Exception('Unexpected heatmap algorithm:', heatmap_algorithm)

    time_stop = perf_counter()
    print('Attenuation map dimensions: ' + str(heatmap_array.shape))
    print('Attenuation map elapsed:', time_stop - time_start)

    print('Building projection array')
    projection_array = library.build_heatmap_max_projection_array(heatmap_array)
    #print(projection_array.shape)

    title = 'Path: ' + file_paths_display_string(file_paths) + '\n'
    title += 'Heatmap Alg: ' + str(heatmap_algorithm)
    title += ',Intensity Dim: ' + str(intensity_array.shape)
    if voxel_dimensions:
        title += ',Voxel Dim: ' + str(voxel_dimensions)

    if view_mean_array:
        view_intensity_array = intensity_mean_array
    else:
        if view_rolled_intensity:
            view_intensity_array = rolled_intensity_array
            surface_positions_for_draw = None
        else:
            view_intensity_array = intensity_array

    attenuation_viewer.view_attenuation(title, view_intensity_array,
        view_intensity_bounds, rolled_intensity_array, heatmap_array,
        heatmap_bounds, projection_array, surface_positions_for_draw)

if __name__ == "__main__":
    main()
