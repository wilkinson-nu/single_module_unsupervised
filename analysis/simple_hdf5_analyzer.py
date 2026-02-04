import h5py
import argparse
import os

def sizeof_fmt(num):
    """Convert bytes to human-readable format."""
    for unit in ['B','KB','MB','GB']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}TB"

def inspect_hdf5_file(filename):
    def visit_func(name, obj):
        if isinstance(obj, h5py.Dataset):
            size = obj.size * obj.dtype.itemsize
            compression = obj.compression or "None"
            print(f"Dataset: {name}")
            print(f"  Shape:       {obj.shape}")
            print(f"  Dtype:       {obj.dtype}")
            print(f"  Size:        {sizeof_fmt(size)}")
            print(f"  Compression: {compression}")
            print()

    print(f"Inspecting: {filename}")
    with h5py.File(filename, 'r') as f:
        f.visititems(visit_func)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect structure and size of datasets in an HDF5 file.")
    parser.add_argument("filename", help="Path to the HDF5 file")
    args = parser.parse_args()

    if not os.path.isfile(args.filename):
        print(f"Error: File '{args.filename}' does not exist.")
        exit(1)

    inspect_hdf5_file(args.filename)
