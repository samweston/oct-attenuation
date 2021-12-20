
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


class AttenuationViewer:
    def __init__(self, title, view_intensity_array, view_intensity_bounds,
            rolled_intensity_array, heatmap_array, heatmap_bounds,
            projection_array, surface_positions_for_draw):
        fig, ax = plt.subplots(2, 2, figsize = (12,6)) # Returns figure and an array of axis
        self.fig = fig
        self.ax = ax
        self.cbar_atten = None

        self.a_scan_axis = self.ax[1][1]
        self.intensity_axis = self.ax[1][0]
        self.projection_axis = self.ax[0][0]
        self.heatmap_axis = self.ax[0][1]

        self.view_intensity_array = view_intensity_array
        self.view_intensity_bounds = view_intensity_bounds
        self.rolled_intensity_array = rolled_intensity_array
        self.heatmap_array = heatmap_array
        self.heatmap_bounds = heatmap_bounds
        self.projection_array = projection_array
        self.surface_positions_for_draw = surface_positions_for_draw

        scroll_cid = fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        click_cid = fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.scan_num = len(view_intensity_array) // 2 # B scan number, start in the middle
        self.a_scan_num = 0 # Index along B scan (a scan number)

        self.title = title
        self.draw_surface_positions = True

        self.update()

    def update(self):
        plt.suptitle(self.title)

        ### Projection Map
        self.update_projection()

        ### Intensity Map
        self.update_intensity()

        ### Attenuation / Heatmap Map
        self.update_heatmap()

        ### Attenuation at A-scan Graph
        self.update_a_scan_attenuation()

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
        index = int((self.scan_num / len(self.view_intensity_array)) * self.projection_array.shape[0])
        xlim = axis.get_xlim()
        axis.plot([xlim[0], xlim[1]], [index, index], 'b-')

    def update_intensity(self):
        b_scan = self.view_intensity_array[self.scan_num]
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

        intensity_axis.set_title('Scan ({}/{})'.format(self.scan_num, len(self.view_intensity_array) - 1))
        intensity_axis.set_xlabel('B-scan length') # (?)
        intensity_axis.set_ylabel('A-scan depth') # (?)

        # Draw the surface
        if self.draw_surface_positions and self.surface_positions_for_draw is not None:
            b_scan_surface_positions = self.surface_positions_for_draw[self.scan_num]
            intensity_axis.plot(range(0, len(b_scan_surface_positions)), b_scan_surface_positions, '-', color='orange')

        # Draw the line showing  the position of the a scan attenuation graph.
        ylim = intensity_axis.get_ylim()
        intensity_axis.plot([self.a_scan_num, self.a_scan_num], [ylim[0], ylim[1]], 'b-')


    def update_heatmap(self):
        heatmap_index = int(self.scan_num * (len(self.heatmap_array) / len(self.view_intensity_array)))
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

    def update_a_scan_attenuation(self):
        a_scan = self.rolled_intensity_array[self.scan_num] # Use the rolled array (use surface).

        self.a_scan_axis.clear()

        # Plot the logarithm line.
        with np.errstate(divide = 'ignore'): # Ignore divide by zeros here.
            y_val = np.log(a_scan[:, self.a_scan_num]) # Takes the log of the intensity values running down the A scan.
        x_val = np.arange(0, np.size(y_val))

        self.a_scan_axis.plot(x_val, y_val)

        # Plot smoothed line using Savitsky Golay filter
        # Not sure what the best parameters for window size and polynomial
        #   order should be. 31 and 3 seem to work alright.
        y_smooth_val = scipy.signal.savgol_filter(y_val, 31, 3)
        self.a_scan_axis.plot(x_val, y_smooth_val)

        # Plot a fit line? (Slope(?))
        draw_fit_line = False
        if draw_fit_line:
            x_range = np.arange(50, 120) # Eh, fitting between these values. Not sure why. Just arbitrary.
            with np.errstate(divide = 'ignore'): # Ignore divide by zeros here.
                fit_curve = np.polyfit(x_range, np.log(a_scan[0:len(x_range), self.a_scan_num]), 1) # Least squares polynomial fit
            self.a_scan_axis.plot(x_range, fit_curve[0] * x_range + fit_curve[1]) # Draw the orange thing, shows part of the polynomial fit curve.
            #atten.append(-10000 * p[0])

        self.a_scan_axis.set_title('Attenuation per A-scan')
        self.a_scan_axis.set_xlabel('A-scan depth')
        self.a_scan_axis.set_ylabel('Log Intensity')

    def show(self):
        plt.show()

    def set_scan_num(self, adjustment):
        self.scan_num = min(len(self.view_intensity_array) - 1, max(0, self.scan_num + adjustment)) # Clamp

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
        if self.intensity_axis.in_axes(event):
            ax_position = self.intensity_axis.transData.inverted().transform((event.x, event.y))
            self.a_scan_num = int(ax_position[0])
            self.update()
        if self.projection_axis.in_axes(event):
            ax_position = self.projection_axis.transData.inverted().transform((event.x, event.y))
            self.scan_num = int((ax_position[1] / self.projection_array.shape[0]) * len(self.view_intensity_array))
            self.update()


def view_attenuation(title, view_intensity_array, view_intensity_bounds,
        rolled_intensity_array, heatmap_array, heatmap_bounds,
        projection_array, surface_positions_for_draw):

    #b_length = 600 * 0.02 # ????
    #depth = 500 * 0.01 / 1.4 # ????

    if view_intensity_array.shape != rolled_intensity_array.shape:
        raise "view_intensity_array.shape != rolled_intensity_array.shape"

    print('There are {} B-scans'.format(view_intensity_array.shape[0]))

    viewer = AttenuationViewer(title, view_intensity_array,
        view_intensity_bounds, rolled_intensity_array, heatmap_array,
        heatmap_bounds, projection_array, surface_positions_for_draw)

    viewer.show()


