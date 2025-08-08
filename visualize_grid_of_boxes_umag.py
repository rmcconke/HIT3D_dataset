import matplotlib.pyplot as plt
import os
import numpy as np  
import pyvista as pv
pv.global_theme.font.family = 'arial'
import tqdm
#pv.global_theme.font.size = 12
from utils import numpy_to_pyvista_velocity_only

def get_timesteps_available(numpy_dir,prefix):
    files = os.listdir(numpy_dir)
    box_files = [f for f in files if f.startswith(prefix+f'_boxnum0') and f.endswith('.npy')]
    timesteps = sorted([(f.split('time')[-1].split('.npy')[0]) for f in box_files])
    return timesteps

def full_filename(numpy_dir, prefix, box_number, timestep):
    return os.path.join(numpy_dir, prefix + f'_boxnum{box_number}_time{timestep}.npy')

def load_box_timeseries(prefix, box_number=1, numpy_dir='numpy_individual_arrays'):
    timesteps = get_timesteps_available(numpy_dir,prefix)
    image_shape = np.load(full_filename(numpy_dir, prefix, box_number, timesteps[0])).shape
    array = np.zeros((len(timesteps),*image_shape))
    for i, timestep in enumerate(timesteps):
        array[i] = np.load(full_filename(numpy_dir, prefix, box_number, timestep))
    return array


scalar_field = 'U_mag'
vmin = 0
vmax = 50
colormap = 'plasma'
n_colors = 32
epsilon = 0.0001
skip_frames = 4
numpy_dir = 'numpy_individual_arrays'
dataset_prefix = 'forced_boxfilter_fs8_highres'

label = r'$|U|$'  

fig, ax = plt.subplots(figsize=(6, 1))
gradient = np.linspace(vmin, vmax, 256).reshape(1, -1)
im = ax.imshow(gradient, aspect='auto', cmap=colormap, extent=[vmin, vmax, 0, 1])
tick_positions = [vmin, (vmin + vmax) / 2, vmax]
tick_labels = [f'{vmin}', f'{(vmin + vmax) / 2:.1f}', f'{vmax}']
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_yticks([])
ax.set_xlabel(label, fontsize=14)
plt.tight_layout()
plt.savefig(f'colourbar_{dataset_prefix}.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

num_boxes = 32 if "decaying" in dataset_prefix else 64
first_data = load_box_timeseries(prefix=dataset_prefix, box_number=0)
num_timesteps = first_data.shape[0]
num_rows = 9
num_cols = int(num_boxes/8)
plotter = pv.Plotter(notebook=False, off_screen=True, shape=(num_rows, num_cols), window_size=[75*num_cols, 75*num_rows])
plotter.open_gif(f'grid_of_boxes_{dataset_prefix}.gif')

timesteps = get_timesteps_available(numpy_dir, dataset_prefix)
timesteps = timesteps[::skip_frames]
for t in tqdm.tqdm(range(len(timesteps))):
    plotter.clear()

    for box_number in range(num_boxes):
        row = box_number // num_cols
        col = box_number % num_cols
        
        plotter.subplot(row, col)
        
        single_timestep_data = np.load(full_filename(numpy_dir, dataset_prefix, box_number, timesteps[t]))
        
        grid = numpy_to_pyvista_velocity_only(single_timestep_data)
        bounds = grid.bounds
        
        slice_data = grid.slice(normal='z', origin=[0, 0, bounds[5] - epsilon])
        plotter.add_mesh(slice_data, scalars=scalar_field, 
                        clim=[vmin, vmax],
                        cmap=colormap,
                        n_colors=n_colors,
                        show_scalar_bar=False,
                        render=False)
        plotter.add_mesh(slice_data.extract_feature_edges(), color='black', line_width=0.5)
        
        slice_data = grid.slice(normal='x', origin=[bounds[1] - epsilon, 0, 0]) 
        plotter.add_mesh(slice_data, scalars=scalar_field, 
                        clim=[vmin, vmax],
                        cmap=colormap,
                        n_colors=n_colors,
                        show_scalar_bar=False,
                        render=False)
        plotter.add_mesh(slice_data.extract_feature_edges(), color='black', line_width=0.5)
        
        slice_data = grid.slice(normal='y', origin=[0, bounds[3] - epsilon, 0])
        plotter.add_mesh(slice_data, scalars=scalar_field, 
                        clim=[vmin, vmax],
                        cmap=colormap,
                        n_colors=n_colors,
                        show_scalar_bar=False,
                        render=False)
        plotter.add_mesh(slice_data.extract_feature_edges(), color='black', line_width=0.5)
        
        plotter.add_text(f'{box_number}', position='lower_left', font_size=10)
    
    plotter.subplot(8, 0)
    plotter.add_text(f't: {timesteps[t]}', position='lower_left', font_size=14)
    
    plotter.write_frame()

plotter.close()


