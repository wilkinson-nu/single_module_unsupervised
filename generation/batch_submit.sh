#!/bin/bash

## Control the number of jobs to spawn
FIRST_JOB=400
LAST_JOB=999

## DUNE FLUX
#EXPT_NAME="DUNEND"
#FLUX_FILE="DUNE_OptimizedEngineeredNov2017_REGULAR.root"
#FLUX_HIST="numu_NDFHC_flux"

## NuMI ME flux
EXPT_NAME="NuMIME"
FLUX_FILE="MINERvA_flux_ME1F.root"
FLUX_HIST="flux_E_cvweighted_CV_WithStatErr"

TARG="1000180400[1.00]"
NU_PDG=14
E_MIN=0.1
E_MAX=50.0
GEOM=argon_box_3m.gdml
NEVENTS=25000
EDEP_MAC=edep.mac

GEN_NAME=GENIE10c
TUNE=G18_10c_00_000
TEMPLATE=batch_GENIEv3_${TUNE}_EDEPSIM_2D_TEMPLATE.sh

## image making
IMAGE_SIZE=512
EXIT_DOWNSTREAM=1
MIN_HITS=10
THRESHOLD=0

## Directory to save to
OUTDIR_ROOT="/global/cfs/cdirs/dune/users/cwilk/nularbox_simulation/GENIEv3_${TUNE}_${EXPT_NAME}"

## Loop over jobs
for N in $(seq ${FIRST_JOB} ${LAST_JOB})
do
    printf -v PADJOB "%03d" ${N}

    OUTFILE_ROOT=nularbox_${EXPT_NAME}_${NU_PDG}_${GEN_NAME}_${PADJOB}

    echo "Processing ${OUTFILE_ROOT}"

    ## Copy the template
    THIS_TEMP=${TEMPLATE/_TEMPLATE/_${PADJOB}}
    cp ${TEMPLATE} ${THIS_TEMP}

    ## Pass on the job specific information
    sed -i "s/__SEED__/${RANDOM}/g" ${THIS_TEMP}
    sed -i "s/__OUTDIR_ROOT__/${OUTDIR_ROOT//\//\\/}/g" ${THIS_TEMP}
    sed -i "s/__OUTFILE_ROOT__/${OUTFILE_ROOT}/g" ${THIS_TEMP}
    sed -i "s/__FLUX_FILE__/${FLUX_FILE}/g" ${THIS_TEMP}
    sed -i "s/__FLUX_HIST__/${FLUX_HIST}/g" ${THIS_TEMP}
    sed -i "s/__TARG__/${TARG}/g" ${THIS_TEMP}
    sed -i "s/__NU_PDG__/${NU_PDG}/g" ${THIS_TEMP}
    sed -i "s/__E_MIN__/${E_MIN}/g" ${THIS_TEMP}
    sed -i "s/__E_MAX__/${E_MAX}/g" ${THIS_TEMP}
    sed -i "s/__GEOM__/${GEOM}/g" ${THIS_TEMP}
    sed -i "s/__NEVENTS__/${NEVENTS}/g" ${THIS_TEMP}
    sed -i "s/__EDEP_MAC__/${EDEP_MAC}/g" ${THIS_TEMP}
    sed -i "s/__IMAGE_SIZE__/${IMAGE_SIZE}/g" ${THIS_TEMP}
    sed -i "s/__EXIT_DOWNSTREAM__/${EXIT_DOWNSTREAM}/g" ${THIS_TEMP}
    sed -i "s/__MIN_HITS__/${MIN_HITS}/g" ${THIS_TEMP}
    sed -i "s/__THRESHOLD__/${THRESHOLD}/g" ${THIS_TEMP}
    
    echo "Submitting ${THIS_TEMP}"

    ## Submit the template
    sbatch ${THIS_TEMP}

    ## No need to delete, so done
    rm ${THIS_TEMP}
done
