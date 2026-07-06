import os
import numpy as np
import h5py
from glob import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

def convert_file(in_path, out_path):
    with h5py.File(in_path, 'r', libver='latest') as fin:
        N = int(fin.attrs['N'])

        # Accumulators for each projection
        xyz_data, xyz_coords = [], []
        xz_data, xz_row, xz_col = [], [], []
        xy_data, xy_row, xy_col = [], [], []
        labels = []
        event_ids = []

        # Offsets start at 0
        xyz_off = np.zeros(N + 1, dtype=np.int64)
        xz_off  = np.zeros(N + 1, dtype=np.int64)
        xy_off  = np.zeros(N + 1, dtype=np.int64)

        # Iterate real groups in numeric order and assert count matches N
        keys = sorted((k for k in fin.keys()), key=lambda s: int(s))
        assert len(keys) == N, f"{in_path}: found {len(keys)} groups but attrs['N']={N}"

        for i, k in enumerate(keys):
            assert int(k) == i, f"{in_path}: non-contiguous group index {k} at position {i}"
            g = fin[k]

            d_xyz = g['data_xyz'][:]
            c_xyz = g['coords_xyz'][:]          # shape (n, 3)
            d_xz  = g['data_xz'][:]
            r_xz  = g['row_xz'][:]
            c_xz  = g['col_xz'][:]
            d_xy  = g['data_xy'][:]
            r_xy  = g['row_xy'][:]
            c_xy  = g['col_xy'][:]

            # Length consistency checks per projection
            assert len(d_xyz) == len(c_xyz), f"{in_path}[{k}]: xyz length mismatch"
            assert len(d_xz) == len(r_xz) == len(c_xz), f"{in_path}[{k}]: xz length mismatch"
            assert len(d_xy) == len(r_xy) == len(c_xy), f"{in_path}[{k}]: xy length mismatch"

            xyz_data.append(d_xyz);  xyz_coords.append(c_xyz)
            xz_data.append(d_xz);    xz_row.append(r_xz);   xz_col.append(c_xz)
            xy_data.append(d_xy);    xy_row.append(r_xy);   xy_col.append(c_xy)

            xyz_off[i + 1] = xyz_off[i] + len(d_xyz)
            xz_off[i + 1]  = xz_off[i]  + len(d_xz)
            xy_off[i + 1]  = xy_off[i]  + len(d_xy)

            labels.append(g['label'][()])
            event_ids.append(np.uint32(g.attrs.get('event_id', i)))

        # Concatenate (handle empty file / empty events gracefully)
        def cat1(chunks, dtype):
            return np.concatenate(chunks).astype(dtype) if chunks else np.zeros(0, dtype=dtype)
        def cat2(chunks, dtype, cols):
            nonempty = [c for c in chunks if c.size]
            return (np.concatenate(nonempty).astype(dtype) if nonempty
                    else np.zeros((0, cols), dtype=dtype))

        xyz_data_a   = cat1(xyz_data, np.float32)
        xyz_coords_a = cat2(xyz_coords, np.uint16, 3)
        xz_data_a = cat1(xz_data, np.float32)
        xz_row_a  = cat1(xz_row,  np.uint16)
        xz_col_a  = cat1(xz_col,  np.uint16)
        xy_data_a = cat1(xy_data, np.float32)
        xy_row_a  = cat1(xy_row,  np.uint16)
        xy_col_a  = cat1(xy_col,  np.uint16)

        labels_a    = np.array(labels, dtype=fin['0']['label'].dtype) if N else np.zeros(0)
        event_id_a  = np.array(event_ids, dtype=np.uint32)

        # Grab file-level / promoted attrs from the source
        shape_3d = fin['0'].attrs['shape_3d'] if N else None
        shape_xz = fin['0'].attrs['shape_xz'] if N else None
        shape_xy = fin['0'].attrs['shape_xy'] if N else None
        src_attrs = dict(fin.attrs)

    # Write output
    tmp_path = out_path + '.tmp'
    with h5py.File(tmp_path, 'w', libver='latest') as fout:
        fout.attrs['N'] = N
        # Preserve schema / enums
        for key in ('label_dtype', 'Topology_enum', 'Mode_enum'):
            if key in src_attrs:
                fout.attrs[key] = src_attrs[key]
        if shape_3d is not None: fout.attrs['shape_3d'] = shape_3d
        if shape_xz is not None: fout.attrs['shape_xz'] = shape_xz
        if shape_xy is not None: fout.attrs['shape_xy'] = shape_xy

        ## Can play with compression
        cargs = {} #dict(compression='gzip', compression_opts=1, shuffle=True)

        fout.create_dataset('xyz_data',   data=xyz_data_a, **cargs)
        fout.create_dataset('xyz_coords', data=xyz_coords_a, **cargs)
        fout.create_dataset('xyz_offsets', data=xyz_off)
        fout.create_dataset('xz_data', data=xz_data_a, **cargs)
        fout.create_dataset('xz_row',  data=xz_row_a, **cargs)
        fout.create_dataset('xz_col',  data=xz_col_a, **cargs)
        fout.create_dataset('xz_offsets', data=xz_off)
        fout.create_dataset('xy_data', data=xy_data_a, **cargs)
        fout.create_dataset('xy_row',  data=xy_row_a, **cargs)
        fout.create_dataset('xy_col',  data=xy_col_a, **cargs)
        fout.create_dataset('xy_offsets', data=xy_off)
        if N: fout.create_dataset('labels', data=labels_a)
        fout.create_dataset('event_id', data=event_id_a)

    os.replace(tmp_path, out_path)  # atomic; avoids half-written files on crash

def _worker(paths):
    """Top-level function so it's picklable. Returns (in_path, status, msg)."""
    in_path, out_path = paths
    try:
        convert_file(in_path, out_path)
        return (in_path, 'ok', None)
    except Exception as e:
        # Clean up any partial temp file
        tmp = out_path + '.tmp'
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass
        return (in_path, 'error', repr(e))

def convert_dir(in_dir, out_dir, workers):
    os.makedirs(out_dir, exist_ok=True)

    jobs = []
    for in_path in sorted(glob(os.path.join(in_dir, '*.h5'))):
        out_path = os.path.join(out_dir, os.path.basename(in_path))
        if os.path.exists(out_path):
            print("skip (exists):", out_path); continue
        jobs.append((in_path, out_path))

    if not jobs:
        print("nothing to do")
        return

    print(f"submitting {len(jobs)} files across {workers} workers")
    done = errors = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker, j): j[0] for j in jobs}
        for fut in as_completed(futures):
            in_path, status, msg = fut.result()
            if status == 'ok':
                done += 1
                print(f"[{done + errors}/{len(jobs)}] ok: {in_path}")
            else:
                errors += 1
                print(f"[{done + errors}/{len(jobs)}] ERROR {in_path}: {msg}")

    print(f"finished: {done} ok, {errors} errors")

if __name__ == '__main__':

    ## Parse some args
    parser = argparse.ArgumentParser("File converter")

    # Require an input file name and location to dump plots
    parser.add_argument('--indir', type=str)
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--workers', type=int, default=4)
    
    # Parse arguments from command line
    args = parser.parse_args()

    ## Report arguments
    for arg in vars(args): print(arg, getattr(args, arg))

    convert_dir(args.indir, args.outdir, args.workers)
