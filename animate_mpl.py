import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import tqdm
import os
# Find and sort all DNS files
filenames = sorted(glob.glob("decaying_hit_box_8_64/Boxes8/DNS/Boxes8_output_*.h5"))
filenames = [f for f in filenames if "DownsampledFiltered" not in f]
filenames = filenames[0:10]
print(f"Found {len(filenames)} files")
box_idx = 0

# Load data for all timesteps
data = []
for filename in tqdm.tqdm(filenames):
    with h5py.File(filename, "r") as f:
        u = f["flow/U"][:]  # (64, 64, 64, 64) = (num_boxes, x, y, z)
        v = f["flow/V"][:]
        w = f["flow/W"][:]
        
        # Take first 64 boxes (for 8x8 grid) and middle z-slice
        mid_z = 32
        timestep_data = []
        u_slice = u[box_idx, :, :, mid_z]  # (64, 64)
        v_slice = v[box_idx, :, :, mid_z]
        w_slice = w[box_idx, :, :, mid_z]
        
        # Compute velocity magnitude
        velocity_mag = np.sqrt(u_slice**2 + v_slice**2 + w_slice**2)
        timestep_data.append(velocity_mag)
    
        data.append(timestep_data)

print(f"Loaded {len(data)} timesteps, each with 64 boxes")

# Create 8x8 subplot animation
fig, axes = plt.subplots(1, 1, figsize=(16, 16))
fig.suptitle('Turbulence Animation - 8x8 Grid of Individual Boxes', fontsize=16)

# Initialize images and find global color scale


vmin = 0
vmax = 50
output_dir = "turbulence_frames_mpl"
os.makedirs(output_dir, exist_ok=True)
# Create figure once
fig, axes = plt.subplots(1, 1, figsize=(16, 16))

# Save each frame as PNG
print("Saving frames as PNG files...")
for frame in tqdm.tqdm(range(len(data))):
    # Clear the axes
    axes.clear()
    
    # Plot the current frame
    im = axes.imshow(data[frame][box_idx], cmap='plasma', 
                     aspect='equal', vmin=vmin, vmax=vmax)
    axes.set_title(f'Box {box_idx}', fontsize=12)
    axes.axis('off')
    
    # Set the main title
    fig.suptitle(f'Turbulence Animation - Timestep {frame}/{len(data)-1}', fontsize=16)
    
    # Add colorbar (optional)
    if frame == 0:  # Add colorbar only once
        plt.colorbar(im, ax=axes, label='Velocity Magnitude')
    
    # Save the frame
    filename = os.path.join(output_dir, f'frame_{frame:04d}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')

plt.close()
print(f"Saved {len(data)} frames to '{output_dir}/' directory")
print("Frames saved as 'frame_0000.png', 'frame_0001.png', etc.")