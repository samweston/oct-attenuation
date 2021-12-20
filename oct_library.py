
#cython: language_level=3
#coding=utf-8

import os
import numpy as np
import numpy.ma as ma
import psutil
import pathlib
import re
import math
from nptdms import TdmsFile
import scipy.stats
import tqdm # Progress bar

# Library from physics department containing TDMS code.
import reference_code.tdmsCode as pytdms

def build_intensity_array(raw_array, apply_hanning_window):

    # Resultant array is cut in half (in third dimension) and rotated.
    intensity_array = np.empty((raw_array.shape[0], int(raw_array.shape[2] / 2), raw_array.shape[1]))

    for i in range(0, raw_array.shape[0]):
        b_scan = raw_array[i]

        # Hanning window. Per email: "just a multiplication of the window to the spectra"
        # May not be necessary on the 800 nm system (?)
        if apply_hanning_window:
            for j in range(len(b_scan)):
                b_scan[j] = np.hanning(len(b_scan[j])) * b_scan[j]

        # Absolute(Fourier Transform( B Scan ) ) 
        b_scan = np.fft.fft(b_scan)
        b_scan = np.absolute(b_scan)

        # Log10 (?). Would need to adjust the colour map when visualising
        #b_scan = np.log10(b_scan) # np.log(b_scan);

        #rot = np.rot90(np.sqrt(abs_ch0 ** 2 + abs_ch1 ** 2)[:, 0:int(A_length / 2)], 3) # Including Retardation.

        # Only need half of the resultant array. Also, rotate by 270 degrees (k=3).
        b_scan = np.rot90(b_scan[:, 0:int(len(b_scan[0]) / 2)], k = 3)

        # Should really flip the image too (so as it appears in the same orientation as the labview software).
        b_scan = np.flip(b_scan, 1)

        intensity_array[i] = b_scan

    return intensity_array

def build_intensity_mean_array(intensity_array):
    intensity_mean_array = []
    vox_c = 5 # Voxel dimension (?)

    # Mean of x (and next 4) in first dimension. Size of a voxel?
    # Essentially seems to just create a smaller array (smaller first dimension, compressed). Why though?
    # Just a fast easy way of averaging in a single B scan.
    for x in range(0, intensity_array.shape[0], vox_c):
        mean = np.mean((intensity_array[x:x + vox_c, :, :]), axis = 0) 
        intensity_mean_array.append(mean)

    return np.array(intensity_mean_array)

# Voxel mean array, allows adjustment in all 3 dimensions.
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

def linear_regress_slope(a, b):
    # TODO: seem to get an error (RuntimeWarning: invalid value encountered in double_scalars)
    # TODO: seem to get an error (RuntimeWarning: invalid value encountered in multiply)
    # Probably just related to zero values.
    with np.errstate(invalid = 'ignore'): 
        return ((a * b).mean() - (a.mean() * b.mean())) / ((a ** 2).mean() - (a.mean() ** 2))

# Using pytdms code.
def build_attenuation_map_1(intensity_array):
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

# Should essentially be same algorithm as pytdms. More convenient mechanism
# for adjusting voxel dimensions though
def build_attenuation_map_2(rolled_intensity_array, voxel_dimensions):
    # Input should already have the surface rolled.
    # Input should include the dimensions of the voxels.
    # Down each A scan (?? length) we need to calculate the slope of the log of the intensities.
    #    Then within each voxel, take the average of these slopes.
    # Dimension 3, should be down the A scan

    shape = rolled_intensity_array.shape
    voxel_dimensions = np.array(voxel_dimensions)

    voxel_array = np.empty(shape // voxel_dimensions)

    depth_range = np.arange(0, voxel_dimensions[1])

    progress_bar = tqdm.tqdm(total = shape[0] // voxel_dimensions[0])

    for i in range(0, shape[0] // voxel_dimensions[0]):
        progress_bar.update(1)
        for j in range(0, shape[1] // voxel_dimensions[1]):
            for k in range(0, shape[2] // voxel_dimensions[2]):
                offset_0 = i * voxel_dimensions[0]
                offset_1 = j * voxel_dimensions[1]
                offset_2 = k * voxel_dimensions[2]

                # Run down the depth and take the mean at each "layer".
                # Depth is within the second dimension (dim[1])
                mean_array = []
                for m in range(0, voxel_dimensions[1]):
                    mean_array.append(np.mean(rolled_intensity_array[
                        offset_0 : offset_0 + voxel_dimensions[0],
                        offset_1 + m,
                        offset_2 : offset_2 + voxel_dimensions[2]]))

                # Ignore divide by zeros here. (RuntimeWarning: divide by zero encountered in log)
                with np.errstate(divide = 'ignore'): 
                    log_vals = np.log(mean_array)

                # Find the slope of these log(means) and this is the attenuation in this voxel.
                # This does a least squares fit I believe. Actually this is linear regression,
                # the numpy.polynomial does a least squares fit (?) not sure really tbh.
                # TODO: Change to numpy.polynomial
                #fit = np.polyfit(depth_range, log_vals, 1)
                #slope = fit[0]

                # Thought this might be faster, actually slower
                #linregress_result = scipy.stats.linregress(depth_range, log_vals)
                #slope = linregress_result.slope

                # Use the direct linear regression vector calculation to find the slope. 
                # Faster, but not massively so.
                slope = linear_regress_slope(depth_range, log_vals)

                voxel_array[i, j, k] = slope

    progress_bar.close()
    return voxel_array

# Just another method for calculating slopes. Take the log values down each
# A scan, smooth (using Savitzky Golay filtering), then record the slopes
# between each point. Actually a lot faster than the other method. I guess
# it's because we don't have to do the regression stuff.
# Probably best to set the second dimension to 1 (e.g. (10, 1, 10))
def build_attenuation_map_3(rolled_intensity_array, voxel_dimensions):
    shape = rolled_intensity_array.shape
    voxel_dimensions = np.array(voxel_dimensions)

    result_shape = shape // voxel_dimensions
    result_shape[1] -= 1
    voxel_array = np.empty(result_shape)

    x_vals = np.arange(0, shape[1] // voxel_dimensions[1])

    with tqdm.tqdm(total = result_shape[0]) as progress_bar:
        for i_0 in range(0, shape[0] // voxel_dimensions[0]):
            progress_bar.update(1)
            for i_2 in range(0, shape[2] // voxel_dimensions[2]):
                offset_0 = i_0 * voxel_dimensions[0]
                offset_2 = i_2 * voxel_dimensions[2]

                # These are our values running down the A scan.
                mean_y_vals = []

                for i_1 in range(0, shape[1] // voxel_dimensions[1]):
                    offset_1 = i_1 * voxel_dimensions[1]
                    mean_y_vals.append(np.mean(rolled_intensity_array[
                        offset_0 : offset_0 + voxel_dimensions[0],
                        offset_1 : offset_1 + voxel_dimensions[1],
                        offset_2 : offset_2 + voxel_dimensions[2]]))

                with np.errstate(divide = 'ignore'):
                    log_y_vals = np.log(mean_y_vals)

                smooth_y_vals = scipy.signal.savgol_filter(log_y_vals, 31, 3)

                # Populate the resultant voxel_array with the slopes.
                for i_1 in range(0, (shape[1] // voxel_dimensions[1]) - 1):
                    with np.errstate(invalid = 'ignore'):
                        voxel_array[i_0, i_1, i_2] = (
                            smooth_y_vals[i_1 + 1] - smooth_y_vals[i_1])

    return voxel_array

# Just find the maximum heatmap value down each projection.
def build_heatmap_max_projection_array(heatmap_array):
    result = np.empty((heatmap_array.shape[0], heatmap_array.shape[2]))

    # Probably a better numpy way of writing this I would think.
    for i in range(0, heatmap_array.shape[0]):
        for j in range(0, heatmap_array.shape[2]):
            max_atten = math.inf

            # FIXME: Run from 5% in, to 2/3 down, bit hacky, could work out where the surface was rolled.
            # Maybe the depth thing in the example code stores this information. That would make sense.
            # Yes, should have surface and depth. Should definitely pass this in.
            for k in range(int(heatmap_array.shape[1] * 0.05), int(heatmap_array.shape[1] * (2.0 / 3.0))):
                max_atten = min(max_atten, heatmap_array[i, k, j])
            result[i, j] = max_atten

    return result

# TODO: Idea is to trace down from the top, find the maximum slope (differential).
# Surely this is the best way of finding an attenuation drop off.
# Doesn't require building of pre-existing array. Less memory usage etc. I guess.
def build_max_differential_projection_array(intensity_array):
    pass


# TODO: Calculate using the formula Abi used. See TDMSProcessBrainAbi.py - 
# I don't really see what this is doing though, the resultant image is weird.
def build_projection_array(intensity_array):
    pass


# Remove the noise at the top of the intensity array.
def remove_top_noise(intensity_array):
    # Another method I think would be to find the depth that contains values > mean. And strip these.
    # Would mean we aren't stripping stuff if there isn't any noise at the top.
    # Should also just apply this to every scan once I've done that.
    return intensity_array[:, 20:, :]

# Surface detection, Rolls stuff at the top to the bottom.
# In place, will replace intensity_array values.
def surface_roll(intensity_array, threshold):
    for b_scan in intensity_array:
        # This function defines surface as the point where "length" number of values (starting at index 0)
        #     have exceeded the threshold.
        # Strange really, would prefer something else (surface is the first dark/high intensity area, 
        # could do something with that).
        surface = np.array(pytdms.surface_detect(b_scan, threshold = threshold, length = 5, skip = 5))

        for y in range(len(surface)):
            b_scan[0 : surface[y], y] = 0 # set the stuff above the surface as zero.
            b_scan[:, y] = np.roll(b_scan[:, y], -1 * surface[y])

def find_surface(intensity_array, threshold):
    shape = intensity_array.shape
    surface_positions = np.empty((shape[0], shape[2]), dtype = np.int32)

    length = 5 # TODO: What is this?
    skip = 5   # TODO: What is this?

    with tqdm.tqdm(total = len(intensity_array)) as progress_bar:
        for i, b_scan in enumerate(intensity_array):
            progress_bar.update(1)
            surface = np.array(pytdms.surface_detect(b_scan,
                threshold = threshold, length = length, skip = skip), dtype = np.int32)

            surface_positions[i] = surface

    # Second positions are for drawing, this is not offset by a single step.
    return surface_positions, surface_positions - (length + skip)

def build_rolled_intensity_array(intensity_array, surface_positions):
    rolled_intensity_array = np.empty(intensity_array.shape)

    for i, b_scan in enumerate(intensity_array):
        surface = surface_positions[i]

        for y in range(len(surface)):
            depth = len(b_scan) - surface[y]
            rolled_intensity_array[i, 0 : depth, y] = b_scan[surface[y] : , y]

    return rolled_intensity_array

# Find surface and depth. Depth will be useful when generating attenuation heatmap. 
# Depth is just a_scan_length - surface[a_scan_num]
def find_surface_and_depth(intensity_array, threshold):

    # Ideally would like to use the neighbouring A scans to improve accuracy of surface.
    # Could even look in 2 dimensions.
    # Really, The surface is the area where the intensity increases rapidly.
    # Could use linear regression (?). Maybe is also what we should be using in the voxel 
    # attenuation calculation, may be a lot faster than that polynomial calculation
    pass

def detect_glass(intensity_array):
    # Ideally would work out a method for detecting glass. Seems to be the area with multiple,
    # very flat areas. These show sharp changes in intensity.
    # How would I do this? Trace down, if there is an area which matches this description,
    # then expand out while there is a similar pattern.
    # The intensity of glass has a sharp rise, then a sharp drop. Could consider this as well.
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
        # it receives twice the signal (a linear relationship). However, thats not how 
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

