import matplotlib.pyplot as plt
import os
import numpy as np  
import pyvista as pv
from utils import load_box_timeseries, numpy_to_pyvista_all_fields

# Animation parameters
scalar_field = 'U_mag'  # Change this to any field you want
vmin = 0
vmax = 50
colormap = 'plasma'
n_colors = 32

data = load_box_timeseries(prefix = 'decaying_boxfilter_fs8_highres', box_number=0)

plotter = pv.Plotter(notebook=False,off_screen=True,window_size=[700, 300])
plotter.open_gif('turbulence_animation.gif')
u_mean = np.mean(data[:, 0], axis=0)
v_mean = np.mean(data[:, 1], axis=0)
w_mean = np.mean(data[:, 2], axis=0)


epsilon = 0.0001 # small offset so slices are just inside the box

for t in range(data.shape[0]):
    plotter.clear()
    grid = numpy_to_pyvista_all_fields(data[t], u_mean=u_mean, v_mean=v_mean, w_mean=w_mean)
    bounds = grid.bounds
    
    slice_data = grid.slice(normal='z', origin=[0, 0, bounds[5] - epsilon])
    plotter.add_mesh(slice_data, scalars=scalar_field, 
                    clim=[vmin, vmax],
                    cmap=colormap,
                    n_colors=n_colors,
                    show_scalar_bar=False,
                    render=False,
                    )
    plotter.add_mesh(slice_data.extract_feature_edges(), color='black', line_width=.5)
    slice_data = grid.slice(normal='x', origin=[bounds[1] - epsilon, 0, 0]) 
    plotter.add_mesh(slice_data, scalars=scalar_field, 
                    clim=[vmin, vmax],
                    cmap=colormap,
                    n_colors=n_colors,
                    show_scalar_bar=False,
                    render=False,
                    )
    plotter.add_mesh(slice_data.extract_feature_edges(), color='black', line_width=.5)
    
    slice_data = grid.slice(normal='y', origin=[0, bounds[3] - epsilon, 0])
    plotter.add_mesh(slice_data, scalars=scalar_field, 
                    clim=[vmin, vmax],
                    cmap=colormap,
                    n_colors=n_colors,
                    show_scalar_bar=False,
                    render=False,
                    )
    plotter.add_mesh(slice_data.extract_feature_edges(), color='black', line_width=.5)
    
    plotter.add_text(f'Time: {t}', position='upper_left')
    
    plotter.write_frame()
    
plotter.close()
print("Animation saved as turbulence_animation.gif")