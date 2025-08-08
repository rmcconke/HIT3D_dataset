import glob
import os
import h5py
import numpy as np
import tqdm

output_folder = 'numpy_individual_arrays'

folders = [
    {'folder': 'decaying_hit_box_8_64/Boxes8/DNS', 'prefix': 'decaying_boxfilter_fs8_highres'},
    {'folder': 'decaying_hit_box_8_64/Boxes8/LES', 'prefix': 'decaying_boxfilter_fs8_lowres'},
    {'folder': 'forced_hit_box_8_64/Boxes8/DNS', 'prefix': 'forced_boxfilter_fs8_highres'},
    {'folder': 'forced_hit_box_8_64/Boxes8/LES', 'prefix': 'forced_boxfilter_fs8_lowres'},
]


def process_folder(folder, prefix):
    files = glob.glob(os.path.join(folder, '*.h5'))
    
    with h5py.File(files[0], 'r') as f:
        n_boxes = f['flow/U'].shape[0]

    for file in tqdm.tqdm(files, desc=f"Processing {prefix}"):
        with h5py.File(file, 'r') as f:
            for box in range(f['flow/U'].shape[0]):
                time = file.split('_')[-1].split('.h5')[0]
                numpy_output = np.empty((3, f['flow/U'].shape[1], f['flow/U'].shape[2], f['flow/U'].shape[3]))
                numpy_output[0,:,:,:] = f['flow/U'][box,:,:,:]
                numpy_output[1,:,:,:] = f['flow/V'][box,:,:,:]
                numpy_output[2,:,:,:] = f['flow/W'][box,:,:,:]
                np.save(os.path.join(output_folder, f'{prefix}_boxnum{box}_time{time}.npy'), numpy_output.astype(np.float32))


if os.path.exists(output_folder):
    for file in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, file))
else:
    os.makedirs(output_folder)

for folder in folders:
    process_folder(folder['folder'], folder['prefix'])
