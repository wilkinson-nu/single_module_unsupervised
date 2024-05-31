import sys
import h5py
from glob import glob

def merge_images(input_file_names, output_file_name, max_images=None):

    outfile = h5py.File(output_file_name, 'w')

    ## Global index for the number of images in the output file
    index = 0

    ## Loop over the input files
    for input_file_name in glob(input_file_names):
        input_file = h5py.File(input_file, 'r')
        
        # Iterate over all event groups in the current input file
        for group in input_file.keys():

            ## Control the number of images to store in the file
            if max_images and index == max_images: break
            
            if group.startswith('event_'):
                # Copy each event group to the new file with a new sequential index
                new_group_name = 'event_'+str(index)
                input_file.copy(group, outfile, new_group_name)
                index += 1

        input_file.attrs['nevents'] = event_index
        ## Close this input file
        input_file.close()

    ## Close the output file, job done
    outfile.close()
    

if __name__ == '__main__':

    ## Take an input file and convert it to an h5 file of images
    if len(sys.argv) < 3:
        print("This script requires: input file list, output file name, [optional: nimages to include]")
        sys.exit()

    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    nimages = None
    if len(sys.argv) == 3: nimages = int(sys.argv[3])
    merge_images(input_file_name, output_file_name, nimages)
