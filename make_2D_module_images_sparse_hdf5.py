import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import coo_matrix

show_plots = False

y_limits = (-61.85430145263672, 61.85430145263672)
z_limits = (-30.816299438476562, 30.816299438476562)

## For v5 reflow or below, must use:
# y_limits = (-83.67790222167969, 40.03070068359375)

def make_image(these_hits):

    ## coo_matrix sums any duplicated values, so don't sum them
    ## These could be extracted from the hits without a loop... I think
    y_list = []
    z_list = []
    q_list = []

    for hit in these_hits:
        
        ## Check for NaNs
        if np.isnan(hit['Q']) or \
           np.isnan(hit['y']) or \
           np.isnan(hit['z']):
            continue

        ## Remove negative charges
        if hit['Q'] < 0: continue
        
        ## Get pixel numbers and charge values
        pixel_pitch = 0.4434
        this_z = int((hit['z'] - z_limits[0])/0.4424 + pixel_pitch/10.)

        pixel_pitch = 0.4434
        this_y = int((hit['y'] - y_limits[0])/0.4424 + pixel_pitch/10.)
        
        this_q = hit['Q']

        y_list.append(this_y)
        z_list.append(this_z)
        q_list.append(this_q)
        
    y_arr = np.array(y_list)
    z_arr = np.array(z_list)
    q_arr = np.array(q_list)

    this_sparse = coo_matrix((q_arr, (y_arr, z_arr)), dtype=np.float32, shape=(280, 140))

    ## Slightly confusing as this isn't necessary for csr or csc matrices
    this_sparse.sum_duplicates()

    # Apply the log10(1+x) transformation to the data array
    trans_data = np.log10(1 + this_sparse.data)

    # Reconstruct the coo_matrix with the transformed data
    trans_sparse = coo_matrix((trans_data, (this_sparse.row, this_sparse.col)), shape=this_sparse.shape)
    
    return trans_sparse


## Check whether there is something in the central region
def filter_central_region(sparse):

    x_min = (140-128)/2
    x_max = x_min+128

    y_min = (280-256)/2
    y_max = y_min+256
    
    x_indices = sparse.col
    y_indices = sparse.row

    # Check if any values are within the specified x and y ranges
    in_x_range = (x_indices >= x_min) & (x_indices <= x_max)
    in_y_range = (y_indices >= y_min) & (y_indices <= y_max)
    
    if np.any(in_x_range & in_y_range): return True
    
    return False


def make_images(input_file_name, output_file_name):

    f = h5py.File(input_file_name, "r")

    ## Allows a minimum number of hits to be an interesting event
    min_hits = 1
    
    raw_events = f['charge/events/data']
    events = raw_events[raw_events['nhit'] > min_hits]

    ## raw_events['id'] for the unique event ID
    
    ## How many events do we have?
    nevts = len(events)
    print("Found:", len(events), "events")
    
    hits = f['charge/calib_prompt_hits/data']
    hits_ref = f['charge/events/ref/charge/calib_prompt_hits/ref']
    hits_region = f['charge/events/ref/charge/calib_prompt_hits/ref_region']

    print("With total:", len(hits), "hits")
    
    if show_plots:
        plt.ion()
        plt.figure(figsize=(4.5, 7))

    sparse_image_list = []
    event_id_list = []

    ## Loop over events
    for evt in range(nevts):

        ## Check on the progress
        if evt % int(nevts/10) == 0 and evt != 0: print("Processed evt", evt, "/", nevts)
        
        ## Now grab all the hits associated with the event
        ev_id = events[evt]['id']
        
        hit_ref = hits_ref[hits_region[ev_id,'start']:hits_region[ev_id,'stop']]
        hit_ref = np.sort(hit_ref[hit_ref[:,0] == ev_id, 1])
        these_hits = hits[hit_ref]

        ## Get an image with 140x280 pixels
        this_sparse = make_image(these_hits)

        ## Check whether this is a "good image"
        if np.count_nonzero(this_sparse.data) > 4000: continue
        if np.count_nonzero(this_sparse.data) < 200: continue

        if not filter_central_region(this_sparse): continue
        
        sparse_image_list .append(this_sparse)
        event_id_list     .append(ev_id)

        if show_plots:
            ## Show image temporarily
            this_image = this_sparse.toarray()
            gr = plt.imshow(this_image, origin='lower')
            plt.colorbar(gr)
            plt.show()

    ## Write the images to an hdf5 file
    with h5py.File(output_file_name, 'w') as fout:

        ## Save the number of images in the file
        fout.attrs['N'] = len(sparse_image_list)

        for i, (sparse_image, event_id) in enumerate(zip(sparse_image_list, event_id_list)):
            group = fout.create_group(str(i))
            group.create_dataset('data', data=sparse_image.data)
            group.create_dataset('row', data=sparse_image.row)
            group.create_dataset('col', data=sparse_image.col)
            group.attrs['shape'] = sparse_image.shape
            group.attrs['event_id'] = event_id
            
    ## Close the input file
    f.close()

if __name__ == '__main__':

    ## Take an input file and convert it to an h5 file of images
    if len(sys.argv) < 3:
        print("An input file and output file name must be provided as arguments!")
        sys.exit()

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    make_images(input_file_name, output_file_name)
