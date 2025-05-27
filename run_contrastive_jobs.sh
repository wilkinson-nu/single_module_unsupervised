#!/bin/bash

# shifter --image=docker:wilkinsonnu/ml_tools:ME python3 single_module_contrastiveONLY_dist_ME.py --indir="/pscratch/sd/c/cwilk/h5_inputs/" --nevents=50000 --lr=5e-4 --latent=128 --hidden_act=silu --latent_act=tanh --ntx_temp=0.5 --nchan=32 --nstep=10 --state_file=test.pth --world_size=4

## These are fixed for now
NSTEP=80
NEVENTS=5000000
NEVTSTRING="5M"

NCHAN=32

batch_size=1536
# batch_size=1024
hidden_act=silu
latent_act=tanh
dropout=0
LATENT=128
aug_type=normaug
scheduler="onecycle"
latent_loss="NTXentMerged"
TEMP=0.5
VERSION=9
INDIR="/pscratch/sd/c/cwilk/h5_inputs_v${VERSION}/"
for LR in 5e-6; do
    for LATENT in 32 64 128; do
	for aug_type in block10x10 bigmodblock10x10; do
	    

	    ROOT_NAME=CONTONLY_lat${LATENT}_nchan${NCHAN}_${LR}_${batch_size}_${latent_loss}${TEMP}_${scheduler}_${aug_type}_${NEVTSTRING}_ME_v${VERSION}
	    
	    LOGFILE=log_contONLY_NEWAUG_260225/log_${ROOT_NAME}
	    
	    ## Spawn a job to process the file
	    JOBSCRIPT=job_${ROOT_NAME}.sh
	    
	    ## The file to save the output into
	    STATE_FILE=state_${ROOT_NAME}.pth
	    
	    echo "#!/bin/bash" > ${JOBSCRIPT}
	    echo "#SBATCH --image=docker:wilkinsonnu/ml_tools:ME" >> ${JOBSCRIPT}
	    echo "#SBATCH --account=dune" >> ${JOBSCRIPT}
	    # echo "#SBATCH --qos=premium" >> ${JOBSCRIPT}
            echo "#SBATCH --qos=regular" >> ${JOBSCRIPT}
	    echo "#SBATCH --constraint=gpu" >> ${JOBSCRIPT}
	    echo "#SBATCH --gpus=4" >> ${JOBSCRIPT}
	    echo "#SBATCH --cpus-per-task=32" >> ${JOBSCRIPT}
	    echo "#SBATCH --time=1440" >> ${JOBSCRIPT}
	    echo "#SBATCH --ntasks-per-node=4" >> ${JOBSCRIPT}

	    # Set environment variables
	    echo "export OMP_NUM_THREADS=16" >> ${JOBSCRIPT}
	    echo "export MASTER_ADDR=\$(hostname)" >> ${JOBSCRIPT}
	    echo "export MASTER_PORT='12355'" >> ${JOBSCRIPT}
	    echo "export WORLD_SIZE=4" >> ${JOBSCRIPT}
	    
	    echo "shifter python3 single_module_contrastive_dist_ME.py \
--indir=${INDIR}      \
--log=${LOGFILE} 	   \
--nevents=${NEVENTS} \
--nstep=${NSTEP} \
--lr=${LR} \
--latent=${LATENT} \
--nchan=${NCHAN} \
--hidden_act=${hidden_act} \
--latent_act=${latent_act} \
--latent_loss=${latent_loss} \
--ntx_temp=${TEMP} \
--batch_size=${batch_size} \
--aug_type=${aug_type} \
--dropout=${dropout} \
--state_file=${SCRATCH}/${STATE_FILE} \
--scheduler=${scheduler} \
--world_size=4" >> ${JOBSCRIPT}
		
	    # Submit job and cleanup
	    sbatch ${JOBSCRIPT}
	    rm ${JOBSCRIPT}
	done
    done
done


