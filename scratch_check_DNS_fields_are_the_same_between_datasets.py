import h5py
import numpy as np
import tqdm
import os

with h5py.File('forced_hit_8_64_subset/DNS/Boxes8_output_0.13950.h5', 'r') as f:
    for box in range(f['flow/U'].shape[0]):
        numpy_output1 = np.empty((3, f['flow/U'].shape[1], f['flow/U'].shape[2], f['flow/U'].shape[3]))
        numpy_output1[0,:,:,:] = f['flow/U'][box,:,:,:]
        numpy_output1[1,:,:,:] = f['flow/V'][box,:,:,:]
        numpy_output1[2,:,:,:] = f['flow/W'][box,:,:,:]
        #np.save(os.path.join(output_folder, f'{prefix}_boxnum{box}_time{time}.npy'), numpy_output.astype(np.float32))

with h5py.File('forced_hit_32_64_subset/DNS/Boxes32_output_0.13950.h5', 'r') as f:
    for box in range(f['flow/U'].shape[0]):
        numpy_output2 = np.empty((3, f['flow/U'].shape[1], f['flow/U'].shape[2], f['flow/U'].shape[3]))
        numpy_output2[0,:,:,:] = f['flow/U'][box,:,:,:]
        numpy_output2[1,:,:,:] = f['flow/V'][box,:,:,:]
        numpy_output2[2,:,:,:] = f['flow/W'][box,:,:,:]

print(numpy_output1[0,0,0,0])
print(numpy_output2[0,0,0,0])

print(np.allclose(numpy_output1, numpy_output2))
print(np.max(np.abs(numpy_output1 - numpy_output2)))