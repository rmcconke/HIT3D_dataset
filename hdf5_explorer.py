import h5py
import os
import argparse

parser = argparse.ArgumentParser(description='Explore HDF5 file')
parser.add_argument('filename', type=str, help='Path to HDF5 file')
args = parser.parse_args()

filename = args.filename

def explore_hdf5(filename):
    # Print file size
    file_size = os.path.getsize(filename)
    if file_size > 1e9:
        print(f"File: {filename} ({file_size/1e9:.1f} GB)")
    elif file_size > 1e6:
        print(f"File: {filename} ({file_size/1e6:.1f} MB)")
    else:
        print(f"File: {filename} ({file_size/1e3:.1f} KB)")
    
    def print_item(name, obj):
        indent = "  " * name.count('/')
        if isinstance(obj, h5py.Group):
            print(f"{indent}Group: {name}/ [{len(obj)} items]")
        else:  # Dataset
            print(f"{indent}Dataset: {name} {obj.shape} {obj.dtype}")
        
        # Print attributes
        for attr_name, attr_val in obj.attrs.items():
            print(f"{indent}  Attribute: {attr_name} = {attr_val}")
    
    with h5py.File(filename, 'r') as f:
        f.visititems(print_item)

explore_hdf5(filename)


