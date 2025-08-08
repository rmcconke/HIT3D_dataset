import os
import numpy as np
import pyvista as pv

def get_timesteps_available(numpy_dir,prefix):
    files = os.listdir(numpy_dir)
    box_files = [f for f in files if f.startswith(prefix+f'_boxnum0') and f.endswith('.npy')]
    timesteps = sorted([(f.split('time')[-1].split('.npy')[0]) for f in box_files])
    return timesteps

def full_filename(numpy_dir, prefix, box_number, timestep):
    return os.path.join(numpy_dir, prefix + f'_boxnum{box_number}_time{timestep}.npy')

def load_box_timeseries(prefix, box_number=1,numpy_dir='numpy_individual_arrays'):
    timesteps = get_timesteps_available(numpy_dir,prefix)
    image_shape = np.load(full_filename(numpy_dir, prefix, box_number, timesteps[0])).shape

    array = np.zeros((len(timesteps),*image_shape))
    for i, timestep in enumerate(timesteps):
        array[i] = np.load(full_filename(numpy_dir, prefix, box_number, timestep))
    return array

def numpy_to_pyvista_velocity_only(data_3d):
    c, l, w, h = data_3d.shape
    grid = pv.ImageData(dimensions=(l+1, w+1, h+1))
    grid.cell_data['U'] = data_3d[0].flatten(order='F')
    grid.cell_data['V'] = data_3d[1].flatten(order='F')
    grid.cell_data['W'] = data_3d[2].flatten(order='F')
    grid.cell_data['U_mag'] = np.linalg.norm(data_3d, axis=0).flatten(order='F')    
    return grid

def numpy_to_pyvista_all_fields(data_3d, u_mean, v_mean, w_mean):
    c, l, w, h = data_3d.shape
    grid = pv.ImageData(dimensions=(l+1, w+1, h+1))
    
    # Instantaneous velocities
    grid.cell_data['U'] = data_3d[0].flatten(order='F')
    grid.cell_data['V'] = data_3d[1].flatten(order='F')
    grid.cell_data['W'] = data_3d[2].flatten(order='F')
    
    # Time-averaged velocities
    grid.cell_data['U_mean'] = u_mean.flatten(order='F')
    grid.cell_data['V_mean'] = v_mean.flatten(order='F')
    grid.cell_data['W_mean'] = w_mean.flatten(order='F')
    
    # Fluctuating components
    u_prime = data_3d[0] - u_mean
    v_prime = data_3d[1] - v_mean
    w_prime = data_3d[2] - w_mean
    
    grid.cell_data['u_prime'] = u_prime.flatten(order='F')
    grid.cell_data['v_prime'] = v_prime.flatten(order='F')
    grid.cell_data['w_prime'] = w_prime.flatten(order='F')
    
    # Velocity magnitudes
    grid.cell_data['U_mag'] = np.linalg.norm(data_3d, axis=0).flatten(order='F')
    grid.cell_data['u_prime_mag'] = np.sqrt(u_prime**2 + v_prime**2 + w_prime**2).flatten(order='F')
      
    # Turbulent kinetic energy (TKE)
    grid.cell_data['k'] = (0.5 * (u_prime**2 + v_prime**2 + w_prime**2)).flatten(order='F')
    
    grid.cell_data['∂U/∂x'] = central_difference_derivative(data_3d[0], axis=0).flatten(order='F')
    grid.cell_data['∂U/∂y'] = central_difference_derivative(data_3d[0], axis=1).flatten(order='F')
    grid.cell_data['∂U/∂z'] = central_difference_derivative(data_3d[0], axis=2).flatten(order='F')

    grid.cell_data['∂V/∂x'] = central_difference_derivative(data_3d[1], axis=0).flatten(order='F')
    grid.cell_data['∂V/∂y'] = central_difference_derivative(data_3d[1], axis=1).flatten(order='F')
    grid.cell_data['∂V/∂z'] = central_difference_derivative(data_3d[1], axis=2).flatten(order='F')

    grid.cell_data['∂W/∂x'] = central_difference_derivative(data_3d[2], axis=0).flatten(order='F')
    grid.cell_data['∂W/∂y'] = central_difference_derivative(data_3d[2], axis=1).flatten(order='F')
    grid.cell_data['∂W/∂z'] = central_difference_derivative(data_3d[2], axis=2).flatten(order='F')

    grid.cell_data['Q'] = compute_q_criterion(grid)
    
    return grid

def central_difference_derivative(field, axis, dx=1):
   return (np.roll(field, -1, axis) - np.roll(field, 1, axis)) / (2 * dx)

def compute_q_criterion(grid):
   # Vorticity tensor components (anti-symmetric part)
   omega_12 = 0.5 * (grid.cell_data['∂U/∂y'] - grid.cell_data['∂V/∂x'])
   omega_13 = 0.5 * (grid.cell_data['∂U/∂z'] - grid.cell_data['∂W/∂x'])
   omega_23 = 0.5 * (grid.cell_data['∂V/∂z'] - grid.cell_data['∂W/∂y'])
   
   # Strain rate tensor components (symmetric part)
   s_11 = grid.cell_data['∂U/∂x']
   s_22 = grid.cell_data['∂V/∂y']
   s_33 = grid.cell_data['∂W/∂z']
   s_12 = 0.5 * (grid.cell_data['∂U/∂y'] + grid.cell_data['∂V/∂x'])
   s_13 = 0.5 * (grid.cell_data['∂U/∂z'] + grid.cell_data['∂W/∂x'])
   s_23 = 0.5 * (grid.cell_data['∂V/∂z'] + grid.cell_data['∂W/∂y'])
   
   # Q = 0.5 * (|Ω|² - |S|²)
   omega_squared = 2 * (omega_12**2 + omega_13**2 + omega_23**2)
   strain_squared = 2 * (s_12**2 + s_13**2 + s_23**2) + s_11**2 + s_22**2 + s_33**2
   
   return 0.5 * (omega_squared - strain_squared)
