import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numba import njit

## This is a pretty ugly function, but... it works
def get_pixel_numz(z_pos):
    z_min = -30.816299438476562   
    pixel_pitch = 0.4434
    z_pixel = int((z_pos - z_min)/0.4424 + pixel_pitch/10.)
    return z_pixel

def get_pixel_numy(y_pos):
    y_min = -83.67790222167969
    pixel_pitch = 0.4434
    y_pixel = int((y_pos - y_min)/0.4424 + pixel_pitch/10.)
    return y_pixel


#filename = "/global/cfs/cdirs/dune/www/data/Module1/TPC12/reflow-test/flowed_v1/packet_2022_02_11_11_39_26_CET_0cd913fb_20220211_113926.data.module1_flow.h5"
filename = "inputs/packet_2022_02_08_07_36_25_CET_0cd913fb_20220208_073625.data.module1_flow.h5"

show_plots = False

@njit
def make_image(these_hits):
    ## Set up an "image" with 140x280 pixels
    this_image = np.zeros((280,140))
    nnan = 0
    for hit in these_hits:
        
        ## Check for NaNs
        if np.isnan(hit['Q']) or \
           np.isnan(hit['y']) or \
           np.isnan(hit['z']):
            nnan += 1
            continue
        
        ## Get pixel numbers and charge values
        z_min = -30.816299438476562   
        pixel_pitch = 0.4434
        this_z = int((hit['z'] - z_min)/0.4424 + pixel_pitch/10.)

        y_min = -83.67790222167969
        pixel_pitch = 0.4434
        this_y = int((hit['y'] - y_min)/0.4424 + pixel_pitch/10.)
        
        this_q = hit['Q']
        
        ## Add to image LarPix(z,y) = image(x,y)
        this_image[this_y, this_z] += this_q
        
    return this_image, nnan

with h5py.File(filename, "r") as f:

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

    ## This is what we'll eventually save
    list_of_images = []
    
    ntotal = 0
    nnan = 0

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

        ## Set up an "image" with 140x280 pixels
        this_image = np.zeros((280,140))

        ntotal += len(these_hits)

        if len(these_hits) == 0:
            print("Skipping!")
            continue
        
        ## Naively think I have to loop over hits
        ## I *think* I can probably use hit_ref to filter these_hits to get a list of all Q that correspond to 
        for hit in these_hits:
        
            ## Check for NaNs
            if np.isnan(hit['Q']) or \
               np.isnan(hit['y']) or \
               np.isnan(hit['z']):
                nnan += 1
                continue
        
            ## Get pixel numbers and charge values
            this_y = get_pixel_numy(hit['y'])
            this_z = get_pixel_numz(hit['z'])
            this_q = hit['Q']
            
            ## Add to image LarPix(z,y) = image(x,y)
            this_image[this_y, this_z] += this_q

        # this_image, this_nnan = make_image(these_hits)
        # nnan += this_nnan
        
        if show_plots:
            ## Show image temporarily
            plt.imshow(this_image, origin='lower')
            # plt.savefig("image"+str(evt)+".png")
            plt.show(block=False)
            input("Continue...")

        list_of_images.append(this_image)

    ## Now report back and save output array
    batch_data  = np.asarray(list_of_images)
    batch_index = np.array(range(len(list_of_images)))
    
    output_file = h5py.File('output_file.h5','w')
    output_file.create_dataset('data',data=batch_data, chunks=tuple([1]+list(batch_data[0].shape)), compression='gzip')
    output_file.create_dataset('index',data=batch_index, chunks=batch_index.shape, compression='gzip')
    output_file.close()

    
    print("Found", nnan, "/", ntotal, "nans")
    
