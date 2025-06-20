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
    
    if show_plots:
        plt.ion()
        plt.figure(figsize=(4.5, 7))

    ## Make output file
    ## Because we already know the number of events at this point, faster to make an empty dataset of the correct size, and just fill it
    output_file = h5py.File(output_file_name,'w')
    blank_image = np.zeros((280,140))
    output_file.create_dataset('data', shape=tuple([nevts]+list(blank_image.shape)), chunks=tuple([1]+list(blank_image.shape)), compression='gzip')
    output_file.create_dataset('index',data=np.array(range(nevts)), chunks=(1,), compression='gzip')
    output_file.close()

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
        this_image = make_image(these_hits)
        
        if show_plots:
            ## Show image temporarily
            plt.imshow(this_image, origin='lower')
            plt.show(block=False)
            input("Continue...")

        ## Append to dataset
        output_file = h5py.File(output_file_name,'a')
        output_file['data'][evt] = this_image        
        output_file.close()
        
    ## End of loop
    print("Finished loop")
    f.close()

if __name__ == '__main__':

    ## Take an input file and convert it to an h5 file of images
    if len(sys.argv) < 3:
        print("An input file and output file name must be provided as arguments!")
        sys.exit()

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    make_images(input_file_name, output_file_name)
