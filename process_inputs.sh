#!/bin/bash

INDIR="/global/cfs/cdirs/dune/www/data/Module1/TPC12/reflow-test/flowed_v1"
OUTDIR="/global/cfs/cdirs/dune/users/cwilk/single_module_images/sparse_joblib_fixdupes_pluscuts"

## Process all files
for INFILEFULL in $(ls ${INDIR}/*); do

    ## SET UP NAMES
    INFILE=${INFILEFULL##*/}
    OUTFILE=${INFILE/.h5/_images.joblib}

    ## Check if the output file exists
    if [ -f "${OUTDIR}/${OUTFILE}" ]; then
        continue
    fi
    echo "Processing ${INFILE}"
    
    ## Spawn a job to process the file
    JOBSCRIPT=jobscript.sh
    
    echo "#!/bin/bash" > ${JOBSCRIPT}
    echo "#SBATCH --image=docker:wilkinsonnu/ml_tools:latest" >> ${JOBSCRIPT}
    echo "#SBATCH --qos=shared" >> ${JOBSCRIPT}
    echo "#SBATCH --constraint=cpu" >> ${JOBSCRIPT}
    echo "#SBATCH --time=120" >> ${JOBSCRIPT}
    echo "#SBATCH --nodes=1" >> ${JOBSCRIPT}
    echo "#SBATCH --ntasks=1" >> ${JOBSCRIPT}
    echo "#SBATCH --mem=4GB" >> ${JOBSCRIPT}
    echo "cd ${SCRATCH}" >> ${JOBSCRIPT}
    echo "shifter python3 ${PWD}/make_2D_module_images_sparse.py ${INDIR}/${INFILE} ${OUTFILE}" >> ${JOBSCRIPT}
    echo "cp ${OUTFILE} ${OUTDIR}/." >> ${JOBSCRIPT}
    echo "rm ${OUTFILE}" >> ${JOBSCRIPT}
    
    # Submit job and cleanup
    sbatch ${JOBSCRIPT}
    rm ${JOBSCRIPT}
done
