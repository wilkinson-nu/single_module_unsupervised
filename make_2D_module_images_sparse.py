import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import coo_matrix
import joblib

show_plots = False

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
        
        ## Get pixel numbers and charge values
        ## (Ugly but it's compiled!)
        z_min = -30.816299438476562   
        pixel_pitch = 0.4434
        this_z = int((hit['z'] - z_min)/0.4424 + pixel_pitch/10.)

        y_min = -83.67790222167969
        pixel_pitch = 0.4434
        this_y = int((hit['y'] - y_min)/0.4424 + pixel_pitch/10.)
        
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
    
    return this_sparse

def make_images(input_file_name, output_file_name):

    f = h5py.File(input_file_name, "r")

    ## Allows a minimum number of hits to be an interesting event
    min_hits = 1
    
    raw_events = f['charge/events/data']
    events = raw_events[raw_events['nhit'] > min_hits]

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

        sparse_image_list .append(this_sparse)
        
        if show_plots:
            ## Show image temporarily
            this_image = this_sparse.toarray()
            plt.imshow(this_image, origin='lower')
            plt.show(block=False)
            input("Continue...")

    ## Joblib is a fast option, but has to load everything into memory... not ideal
    joblib.dump(sparse_image_list, output_file_name)
        
    f.close()

if __name__ == '__main__':

    ## Take an input file and convert it to an h5 file of images
    if len(sys.argv) < 3:
        print("An input file and output file name must be provided as arguments!")
        sys.exit()

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    make_images(input_file_name, output_file_name)
