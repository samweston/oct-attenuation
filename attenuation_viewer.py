
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

MODE_SHOW_ALL, MODE_SHOW_INTENSITY, MODE_SHOW_A_SCAN = 0, 1, 2

class AttenuationViewer:
    def __init__(self, title, mode, view_intensity_array, view_intensity_bounds,
            rolled_intensity_array, heatmap_array, heatmap_bounds,
            projection_array, surface_positions_for_draw):

        self.mode = mode

        if mode == MODE_SHOW_ALL:
            self.fig, self.ax = plt.subplots(2, 2, figsize = (12,6)) # Returns figure and an array of axis
            self.a_scan_axis = self.ax[1][0]
            self.intensity_axis = self.ax[1][1]
            self.projection_axis = self.ax[0][0]
            self.heatmap_axis = self.ax[0][1]
        elif mode == MODE_SHOW_INTENSITY:
            self.fig, self.ax = plt.subplots(1, 2, figsize = (12,6)) # Returns figure and an array of axis
            self.a_scan_axis = self.ax[0]
            self.intensity_axis = self.ax[1]
        elif mode == MODE_SHOW_A_SCAN:
            self.fig, self.ax = plt.subplots(1, 1, figsize = (12,6)) # Returns figure and an array of axis
            self.a_scan_axis = self.ax
        else:
            raise Exception(f'Unexpected mode {mode}')

        self.cbar_atten = None

        self.view_intensity_array = view_intensity_array
        self.view_intensity_bounds = view_intensity_bounds
        self.rolled_intensity_array = rolled_intensity_array
        self.heatmap_array = heatmap_array
        self.heatmap_bounds = heatmap_bounds
        self.projection_array = projection_array
        self.surface_positions_for_draw = surface_positions_for_draw

        scroll_cid = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        click_cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.b_scan_num = len(view_intensity_array) // 2 # B scan number, start in the middle
        self.a_scan_num = 0 # Index along B scan (a scan number)

        self.title = title
        self.draw_surface_positions = True
        self.draw_vertical_a_scan = False

        self.update()

    def update(self):
        plt.suptitle(self.title)

        ### Projection Map
        if self.mode == MODE_SHOW_ALL:
            self.update_projection()

        ### Intensity Map
        if self.mode in [MODE_SHOW_ALL, MODE_SHOW_INTENSITY]:
            self.update_intensity()

        ### Attenuation / Heatmap Map
        if self.mode == MODE_SHOW_ALL:
            self.update_heatmap()

        ### Intensity at A-scan Graph
        if self.mode in [MODE_SHOW_ALL, MODE_SHOW_INTENSITY, MODE_SHOW_A_SCAN]:
            self.update_a_scan_intensity()

        print(f'a_scan={self.a_scan_num}, b_scan={self.b_scan_num}')

        ### Draw
        self.fig.canvas.draw()

    def update_projection(self):
        axis = self.projection_axis
        axis.clear()

        im_projection = axis.imshow(self.projection_array,
            cmap = 'jet_r', interpolation = 'none', aspect = 'auto',
            vmin = self.heatmap_bounds[0], vmax = self.heatmap_bounds[1])

        axis.set_title('Projection')

        # Draw the line showing  the position of the visible b scan.
        # [x1, x2, ...], [y1, y2, ...]
        index = int((self.b_scan_num / len(self.view_intensity_array)) * self.projection_array.shape[0])
        xlim = axis.get_xlim()
        axis.plot([xlim[0], xlim[1]], [index, index], 'b-')

    def update_intensity(self):
        b_scan = self.view_intensity_array[self.b_scan_num]
        intensity_axis = self.intensity_axis

        intensity_axis.clear()

        # 'binary_r' - Flipped
        # This is what Abi used with his power law transform.
#        im_intensity = intensity_axis.imshow(b_scan, cmap = 'Greys_r', interpolation = 'none', aspect = 'auto', vmin = 0, vmax = 2)

        im_intensity = intensity_axis.imshow(b_scan, cmap = 'binary',
            interpolation = 'none', aspect = 'auto',
            vmin = self.view_intensity_bounds[0],
            vmax = self.view_intensity_bounds[1])
            #, extent=[0, b_length, depth, 0])

        intensity_axis.set_title('Scan ({}/{})'.format(self.b_scan_num, len(self.view_intensity_array) - 1))
        intensity_axis.set_title('B-scan Intensity')
        intensity_axis.set_xlabel('B-scan length') # (units=?)
        intensity_axis.set_ylabel('A-scan depth') # (units=?)

        # Draw the surface
        if self.draw_surface_positions and self.surface_positions_for_draw is not None:
            b_scan_surface_positions = self.surface_positions_for_draw[self.b_scan_num]
            intensity_axis.plot(range(0, len(b_scan_surface_positions)), b_scan_surface_positions, '-', color='orange')

        # Draw the line showing  the position of the a scan attenuation graph.
        ylim = intensity_axis.get_ylim()
        intensity_axis.plot([self.a_scan_num, self.a_scan_num], [ylim[0], ylim[1]], 'b-')


    def update_heatmap(self):
        heatmap_index = int(self.b_scan_num * (len(self.heatmap_array) / len(self.view_intensity_array)))
        a_scan_heatmap = self.heatmap_array[heatmap_index]

        self.heatmap_axis.clear()

        im_attenuation = self.heatmap_axis.imshow(a_scan_heatmap,
            cmap = 'jet_r', interpolation = 'none', aspect = 'auto',
            vmin = self.heatmap_bounds[0], vmax = self.heatmap_bounds[1])
            # , extent=[0,b_length,depth,0])

        if self.cbar_atten == None:
            self.cbar_atten = self.fig.colorbar(im_attenuation, ax = self.heatmap_axis) #### make the colorbar
            self.cbar_atten.set_label('Attenuation coefficient', rotation = 270)

        self.heatmap_axis.set_title('Attenuation-Heatmap ({}/{})'.format(heatmap_index, len(self.heatmap_array) - 1))
        self.heatmap_axis.set_xlabel('B-scan Length')
        self.heatmap_axis.set_ylabel('Depth')

    # This could actually include the slope, which would be the attenuation.
    def update_a_scan_intensity(self):
        a_scan = self.rolled_intensity_array[self.b_scan_num] # Use the rolled array (use surface).

        self.a_scan_axis.clear()

        # Plot the logarithm line.
        with np.errstate(divide = 'ignore'): # Ignore divide by zeros here.
            y_val = a_scan[:, self.a_scan_num]
            y_val = y_val[y_val != 0]

            # Take the log (natural) of the intensity values.
            # y_val = np.log(y_val)
        x_val = np.arange(0, np.size(y_val))

        # Plot smoothed line using Savitsky Golay filter
        # Not sure what the best parameters for window size and polynomial
        #   order should be. 31 and 3 seem to work alright.
        y_smooth_val = scipy.signal.savgol_filter(y_val, 31, 3)

        self.a_scan_axis.set_title('A-scan Intensity')
        depth_label = 'Depth'
        intensity_label = 'Intensity'

        if self.draw_vertical_a_scan:
            # Vertical line graph, depth is on the y axis. Just flip the axis.
            self.a_scan_axis.invert_yaxis()
            self.a_scan_axis.set_xscale('log')

            self.a_scan_axis.plot(y_val, x_val)
            self.a_scan_axis.plot(y_smooth_val, x_val, color='red')

            self.a_scan_axis.set_xlabel(intensity_label)
            self.a_scan_axis.set_ylabel(depth_label)
        else:
            # Default view, depth is on the x axis.
            self.a_scan_axis.set_yscale('log')

            self.a_scan_axis.plot(x_val, y_val)
            self.a_scan_axis.plot(x_val, y_smooth_val, color='red')

            self.a_scan_axis.set_xlabel(depth_label)
            self.a_scan_axis.set_ylabel(intensity_label)

        # # Plot a fit line? (Slope(?))
        # draw_fit_line = False
        # if draw_fit_line:
        #     x_range = np.arange(50, 120) # Eh, fitting between these values. Not sure why. Just arbitrary.
        #     with np.errstate(divide = 'ignore'): # Ignore divide by zeros here.
        #         fit_curve = np.polyfit(x_range, np.log(a_scan[0:len(x_range), self.a_scan_num]), 1) # Least squares polynomial fit
        #     self.a_scan_axis.plot(x_range, fit_curve[0] * x_range + fit_curve[1]) # Draw the orange thing, shows part of the polynomial fit curve.
        #     #atten.append(-10000 * p[0])

    def show(self):
        plt.show()

    def set_scan_num(self, adjustment):
        self.b_scan_num = min(len(self.view_intensity_array) - 1, max(0, self.b_scan_num + adjustment)) # Clamp

    def on_scroll(self, event):
        # https://matplotlib.org/stable/gallery/event_handling/image_slices_viewer.html
        adjustment = 1
        if event.key == 'shift':
            adjustment = 5

        if event.button == 'up':
            self.set_scan_num(adjustment)
        else:
            self.set_scan_num(-1 * adjustment)

        self.update()

    def on_click(self, event):
        if self.mode in [MODE_SHOW_ALL, MODE_SHOW_INTENSITY]:
            if self.intensity_axis.in_axes(event):
                ax_position = self.intensity_axis.transData.inverted().transform((event.x, event.y))
                self.a_scan_num = int(ax_position[0])
                self.update()
        if self.mode == MODE_SHOW_ALL:
            if self.projection_axis.in_axes(event):
                ax_position = self.projection_axis.transData.inverted().transform((event.x, event.y))
                self.b_scan_num = int((ax_position[1] / self.projection_array.shape[0]) * len(self.view_intensity_array))
                self.update()


def view_attenuation(title, view_intensity_array, view_intensity_bounds,
        rolled_intensity_array, heatmap_array, heatmap_bounds,
        projection_array, surface_positions_for_draw,
        a_scan_num=None, b_scan_num=None):

    #b_length = 600 * 0.02 # ????
    #depth = 500 * 0.01 / 1.4 # ????

    if view_intensity_array.shape != rolled_intensity_array.shape:
        raise "view_intensity_array.shape != rolled_intensity_array.shape"

    print('There are {} B-scans'.format(view_intensity_array.shape[0]))

    mode = MODE_SHOW_ALL

    viewer = AttenuationViewer(title, mode, view_intensity_array,
        view_intensity_bounds, rolled_intensity_array, heatmap_array,
        heatmap_bounds, projection_array, surface_positions_for_draw)

    if a_scan_num is not None:
        viewer.a_scan_num = a_scan_num
        viewer.update()
    if b_scan_num is not None:
        viewer.b_scan_num = b_scan_num
        viewer.update()

    viewer.show()


