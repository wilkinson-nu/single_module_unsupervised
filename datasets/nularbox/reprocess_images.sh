#!/bin/bash

INPUT_DIR=${PWD}
RUNSCRIPT=make_2D_nusim_images.py

## Input and output directories
INDIR=/global/cfs/cdirs/dune/users/cwilk/nularbox_simulation/EDEPSIM
OUTDIR=/global/cfs/cdirs/dune/users/cwilk/nularbox_simulation/IMAGES2D

for INFILEFULL in $(ls ${INDIR}/*.root); do
    
    INFILE=${INFILEFULL##*/}

    ## Check if the h5 file exists, skip if so
    if [ -f "${OUTDIR}/${INFILE/.root/.h5}" ]; then
        continue
    fi

    ## This is the submission script
    THIS_SUB=${INFILE/.root/.sh}

    echo ${THIS_SUB}
    
    ## Boilerplate
    echo "#!/bin/bash" > ${THIS_SUB}
    echo "#SBATCH --image=docker:wilkinsonnu/simple_det_sim:latest" >> ${THIS_SUB}
    echo "#SBATCH --qos=shared" >> ${THIS_SUB}
    echo "#SBATCH --constraint=cpu" >> ${THIS_SUB}
    echo "#SBATCH --time=60" >> ${THIS_SUB}
    echo "#SBATCH --nodes=1" >> ${THIS_SUB}
    echo "#SBATCH --ntasks=1" >> ${THIS_SUB}
    echo "#SBATCH --mem=4GB" >> ${THIS_SUB}
    
    ## Make the appropriate directories
    TEMP_DIR=${SCRATCH}/${INFILE/.root/}
    echo "echo 'Moving to SCRATCH: ${TEMP_DIR}'" >> ${THIS_SUB}
    echo "mkdir ${TEMP_DIR}" >> ${THIS_SUB}
    echo "cd ${TEMP_DIR}" >> ${THIS_SUB}
    
    ## Get the file to convert
    echo "cp ${INFILEFULL} ${INFILE}" >> ${THIS_SUB}

    ## Copy the python script to run with... also a horrible hack to get other libraries in...
    echo "cp ${INPUT_DIR}/*.py ." >> ${THIS_SUB}
    
    ## Do the business
    echo "shifter python3 ${RUNSCRIPT} ${INFILE} ${INFILE/.root/.h5}" >> ${THIS_SUB}
    echo "cp ${INFILE/.root/.h5} ${OUTDIR}/." >> ${THIS_SUB}

    ## Clean up
    echo "rm -r ${TEMP_DIR}" >> ${THIS_SUB}
    
    echo "Submitting ${THIS_SUB}"
	    
    ## Submit the template
    sbatch ${THIS_SUB}
    
    ## No need to delete, so done
    rm ${THIS_SUB}
    
done
