
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig


class AttenuationViewer:
    def __init__(self, title, intensity_array, heatmap_array, projection_array, surface_positions):
        fig, ax = plt.subplots(2, 2, figsize = (12,6)) # Returns figure and an array of axis
        self.fig = fig
        self.ax = ax
        self.cbar_atten = None
        self.intensity_axis = self.ax[1][0]
        self.projection_axis = self.ax[0][0]
        
        self.intensity_array = intensity_array
        self.heatmap_array = heatmap_array
        self.projection_array = projection_array
        self.surface_positions = surface_positions
        
        scroll_cid = fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        click_cid = fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.scan_num = len(intensity_array) // 2 # B scan number, start in the middle
        self.a_scan_num = 0 # Index along B scan (a scan number)
        
        self.title = title
        
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
        
        im_projection = axis.imshow(self.projection_array, cmap = 'jet_r', interpolation = 'none', aspect = 'auto',
            vmax = 0, vmin= -0.06) # , extent=[0, b_length, depth,0])
        
        axis.set_title('Projection')
            
        # Draw the line showing  the position of the visible b scan.
        # [x1, x2, ...], [y1, y2, ...]
        index = int((self.scan_num / len(self.intensity_array)) * self.projection_array.shape[0])
        xlim = axis.get_xlim()
        axis.plot([xlim[0], xlim[1]], [index, index], 'b-')
            
    def update_intensity(self):
        b_scan = self.intensity_array[self.scan_num]
        intensity_axis = self.intensity_axis
        
        intensity_axis.clear()
    
        # 'binary_r' - Flipped
        # This is what Abi used with his power law transform.
#        im_intensity = intensity_axis.imshow(b_scan, cmap = 'Greys_r', interpolation = 'none', aspect = 'auto', vmin = 0, vmax = 2)

        im_intensity = intensity_axis.imshow(b_scan, cmap = 'binary', interpolation = 'none', aspect = 'auto', 
            vmin = 0, vmax = 20000) #, extent=[0, b_length, depth, 0])
            
        intensity_axis.set_title('Scan ({}/{})'.format(self.scan_num, len(self.intensity_array) - 1))
        intensity_axis.set_xlabel('B-scan length (??)') # (?)
        intensity_axis.set_ylabel('A-scan length (??)') # (?)
        
        # Draw the surface
        b_scan_surface_positions = self.surface_positions[self.scan_num]
        intensity_axis.plot(range(0, len(b_scan_surface_positions)), b_scan_surface_positions, '-', color='orange')
        
        # Draw the line showing  the position of the a scan attenuation graph.
        ylim = intensity_axis.get_ylim()
        intensity_axis.plot([self.a_scan_num, self.a_scan_num], [ylim[0], ylim[1]], 'b-')
        
        
    def update_heatmap(self):
        heatmap_index = int(self.scan_num * (len(self.heatmap_array) / len(self.intensity_array)))
        a_scan_heatmap = self.heatmap_array[heatmap_index]
        a_scan_heatmap_axis = self.ax[0][1]
        
        a_scan_heatmap_axis.clear()

        im_attenuation = a_scan_heatmap_axis.imshow(a_scan_heatmap, cmap = 'jet_r', interpolation = 'none', aspect = 'auto',
            vmax = 0, vmin= -0.06) # , extent=[0,b_length,depth,0])
        if self.cbar_atten == None:
            self.cbar_atten = self.fig.colorbar(im_attenuation, ax = a_scan_heatmap_axis) #### make the colorbar
            self.cbar_atten.set_label('Attenuation coefficient', rotation=270)
            
        a_scan_heatmap_axis.set_title('Attenuation-Heatmap ({}/{})'.format(heatmap_index, len(self.heatmap_array) - 1))
        a_scan_heatmap_axis.set_xlabel('B-scan Length (??)')
        a_scan_heatmap_axis.set_ylabel('Depth (??)')
        
    def update_a_scan_attenuation(self):
        a_scan = self.intensity_array[self.scan_num]
        
        atten_graph_axis = self.ax[1][1]
        
        atten_graph_axis.clear()
       
        # Plot the logarithm line.
        with np.errstate(divide = 'ignore'): # Ignore divide by zeros here.
            yval = np.log(a_scan[:, self.a_scan_num]) # Takes the log of the intensity values running down the A scan.
        atten_graph_axis.plot(np.arange(0, np.size(yval), 1), yval)
        
        # Plot a fit line? (Slope(?))
        draw_fit_line = False
        if draw_fit_line:
            x_range = np.arange(50, 120) # Eh, fitting between these values. Not sure why. Just arbitrary.
            with np.errstate(divide = 'ignore'): # Ignore divide by zeros here.
                fit_curve = np.polyfit(x_range, np.log(a_scan[0:len(x_range), self.a_scan_num]), 1) # Least squares polynomial fit
            atten_graph_axis.plot(x_range, fit_curve[0] * x_range + fit_curve[1]) # Draw the orange thing, shows part of the polynomial fit curve.
            #atten.append(-10000 * p[0])
        
        atten_graph_axis.set_title('Attenuation per A-scan')
        atten_graph_axis.set_xlabel('A-scan length (??)')
        atten_graph_axis.set_ylabel('Log Intensity')
        
    def show(self):
        plt.show()
    
    def set_scan_num(self, adjustment):
        self.scan_num = min(len(self.intensity_array) - 1, max(0, self.scan_num + adjustment)) # Clamp
    
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
            self.scan_num = int((ax_position[1] / self.projection_array.shape[0]) * len(self.intensity_array))
            self.update()


def view_attenuation(title, _intensity_array, _heatmap_array, _projection_array, surface_positions):
    intensity_array = _intensity_array
    
    heatmap_array = _heatmap_array
    
    projection_array = _projection_array

    #b_length = 600 * 0.02 # ????
    #depth = 500 * 0.01 / 1.4 # ????
    
    print('There are {} B-scans'.format(intensity_array.shape[0]))
    
#    while True:
#        a_scan = input("Enter scan to view: ") # A (?), check this.
#        a_scan_num = -1
#        try:
#            a_scan_num = int(a_scan)
#        except ValueError:
#            print("invalid A scan num")
#            break
#            
#        view_a_scan(a_scan_num, intensity_array, heatmap_array)
    
    viewer = AttenuationViewer(title, intensity_array, heatmap_array, projection_array, surface_positions)
    viewer.show()
    


    
    
    
