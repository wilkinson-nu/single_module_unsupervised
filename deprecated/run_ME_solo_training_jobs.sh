#!/bin/bash

## shifter --image=docker:wilkinsonnu/ml_tools:ME python3 single_module_NN_solo_dist_ME.py --indir="/pscratch/sd/c/cwilk/h5_inputs/" --nevents=500000 --lr=5e-4 --latent=32 --nchan=16 --nstep=10 --arch=simple --augment --state_file=test.pth --restart --world_size=4


INDIR="/pscratch/sd/c/cwilk/h5_inputs/"

## These are fixed for now
NSTEP=50
NEVENTS=4000000
NEVTSTRING="4M"
scheduler="onecycle"

NCHAN=16
arch=simple

hidden_act=silu
latent_act=tanh
dropout=0

#for LR in 5e-4 1e-4 2e-5; do #1e-2 1e-3 1e-4 1e-5; do
for LR in 5e-6; do
    #for LATENT in 32 64 192; do
    for LATENT in 128; do
	for AUG_TYPE in unitcharge; do
	    for NCHAN in 32; do
		

		## Make a log file
		ROOT_NAME=lat${LATENT}_nchan${NCHAN}_${LR}_drop${dropout}_arch${arch}_SOLO_actfns_${hidden_act}_${latent_act}_${NEVTSTRING}_${scheduler}_${AUG_TYPE}_ME
		LOGFILE=log_unitcharge_200125/log_${ROOT_NAME}
		
		## Spawn a job to process the file
		JOBSCRIPT=job_${ROOT_NAME}.sh
		
		## The file to save the output into
		STATE_FILE=state_${ROOT_NAME}.pth
		
		echo "#!/bin/bash" > ${JOBSCRIPT}
		echo "#SBATCH --image=docker:wilkinsonnu/ml_tools:ME" >> ${JOBSCRIPT}
		echo "#SBATCH --account=dune" >> ${JOBSCRIPT}
		echo "#SBATCH --qos=regular" >> ${JOBSCRIPT}
		# echo "#SBATCH --qos=premium" >> ${JOBSCRIPT}
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
		
		echo "shifter python3 single_module_NN_solo_dist_ME.py --indir=${INDIR} --nevents=${NEVENTS} --log=${LOGFILE} --lr=${LR} \
		     --latent=${LATENT} --nchan=${NCHAN} --nstep=${NSTEP} --arch=${arch} --aug_type=${AUG_TYPE} \
		     --hidden_act=${hidden_act} --latent_act=${latent_act} --dropout=${dropout} \
		     --state_file=${SCRATCH}/${STATE_FILE} --scheduler=${scheduler} --world_size=4" >> ${JOBSCRIPT}	 

		# Submit job and cleanup
		sbatch ${JOBSCRIPT}
		rm ${JOBSCRIPT}
	    done
	done
    done
done

