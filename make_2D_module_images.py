import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numba import njit

show_plots = False

@njit
def make_image(these_hits):

    ## Set up an "image" with 140x280 pixels
    this_image = np.zeros((280,140))
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
        
        ## Add to image LarPix(z,y) = image(x,y)
        this_image[this_y, this_z] += this_q
        
    return this_image

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
    
    ## This is what we'll eventually save
    list_of_images = []
    
    if show_plots:
        plt.ion()
        plt.figure(figsize=(4.5, 7))

    ## Loop over events
    for evt in range(nevts):
    
        ## Now grab all the hits associated with the event
        ev_id = events[evt]['id']
        
        hit_ref = hits_ref[hits_region[ev_id,'start']:hits_region[ev_id,'stop']]
        hit_ref = np.sort(hit_ref[hit_ref[:,0] == ev_id, 1])
        these_hits = hits[hit_ref]

        ## Get an image with 140x280 pixels
        this_image = make_image(these_hits)
        
        if show_plots:
            ## Show image temporarily
            plt.imshow(this_image, origin='lower')
            plt.show(block=False)
            input("Continue...")

        list_of_images.append(this_image)

    ## Now report back and save output array
    batch_data  = np.asarray(list_of_images)
    batch_index = np.array(range(len(list_of_images)))
    
    output_file = h5py.File(output_file_name,'w')
    output_file.create_dataset('data',data=batch_data, chunks=tuple([1]+list(batch_data[0].shape)), compression='gzip')
    output_file.create_dataset('index',data=batch_index, chunks=batch_index.shape, compression='gzip')
    output_file.close()

    f.close()

if __name__ == '__main__':

    ## Take an input file and convert it to an h5 file of images
    if len(sys.argv) < 3:
        print("An input file and output file name must be provided as arguments!")
        sys.exit()

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    make_images(input_file_name, output_file_name)
