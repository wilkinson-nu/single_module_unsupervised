import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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


filename = "/global/cfs/cdirs/dune/www/data/Module1/TPC12/reflow-test/flowed_v1/packet_2022_02_11_11_39_26_CET_0cd913fb_20220211_113926.data.module1_flow.h5"

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
    
    ntotal = 0
    nnan = 0
    
    ## Loop over events
    for evt in range(nevts):
    
        ## Now grab all the hits associated with the event
        ev_id = events[evt]['id']
        
        hit_ref = hits_ref[hits_region[ev_id,'start']:hits_region[ev_id,'stop']]
        hit_ref = np.sort(hit_ref[hit_ref[:,0] == ev_id, 1])
        these_hits = hits[hit_ref]

        ## Set up an "image" with 140x280 pixels (what format?)
        this_image = np.zeros((140,280))

        ntotal += len(these_hits)

        if len(these_hits) == 0:
            print("Skipping!")
            continue
        
        ## Naively think I have to loop over hits
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
            this_image[this_z, this_y] += this_q
            
        ## Show image temporarily
        plt.imshow(this_image.squeeze(), origin='lower')
        plt.show()
        input("Continue...")
        #print(this_image)
    #print(charge_evt['ref/charge/calib_prompt_hits'][0])
    
    ## Can loop over calib hits like this
    #print(calib_hits['x'][0])
    #ds_arr = calib_hits[()]

    #print(ds_arr[0]['x'])

    print("Found", nnan, "/", ntotal, "nans")
    
    #print(data)
    
    # preferred methods to get dataset values:
    # ds_obj = calib_hits      # returns as a h5py dataset object
    # ds_arr = calib_hits[()]  # returns as a numpy array
