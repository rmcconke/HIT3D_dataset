import matplotlib.pyplot as plt
import os
import numpy as np  
import pyvista as pv
from utils import load_box_timeseries, numpy_to_pyvista_all_fields
import tqdm
import time
# Animation parameters
scalar_field = 'Q'  # Change this to any field you want
isocontour_value = 0.05
color_by = 'U_mag'
vmin = 0
vmax = 50
colormap = 'plasma'
n_colors = 128
skip_frames = 4


data = load_box_timeseries(prefix = 'forced_boxfilter_fs8_highres', box_number=0)

plotter = pv.Plotter(notebook=False,off_screen=True,window_size=[550, 400])
plotter.open_gif('isocontour_q_criterion.gif')
u_mean = np.mean(data[:, 0], axis=0)
v_mean = np.mean(data[:, 1], axis=0)
w_mean = np.mean(data[:, 2], axis=0)


epsilon = 0.0001 # small offset so slices are just inside the box

for t in tqdm.tqdm(range(data.shape[0])[::skip_frames]):
    plotter.clear()
    grid = numpy_to_pyvista_all_fields(data[t], u_mean=u_mean, v_mean=v_mean, w_mean=w_mean)
    #plotter.camera.focal_point = grid.center

    grid = grid.cell_data_to_point_data()

    bounds = grid.bounds
    contour = grid.contour([isocontour_value], scalars=scalar_field)
    #slice_data = grid.slice(normal='z', origin=[0, 0, bounds[5] - epsilon])
    plotter.add_mesh(contour, scalars=color_by, 
                    clim=[vmin, vmax],
                    cmap=colormap,
                    n_colors=n_colors,
                    show_scalar_bar=False,
                    render=True,specular=0.8, specular_power=100, smooth_shading=True,
                    )
    
    plotter.add_mesh(grid.extract_feature_edges(), color='black', line_width=.5)
    plotter.enable_lightkit()  # Adds a default light setup
    plotter.add_text(f'Time: {t}', position='upper_left', font_size=10)
    #plotter.camera.azimuth += 90/len(range(data.shape[0])[::skip_frames])
    plotter.write_frame()
    
plotter.close()
print("Animation saved as turbulence_animation.gif")