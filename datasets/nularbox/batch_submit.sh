#!/bin/bash

## Control the number of jobs to spawn
FIRST_JOB=0
LAST_JOB=9

## Control the jobs
OUTDIR_ROOT="/global/cfs/cdirs/dune/users/cwilk/nularbox_simulation"

FLUX_FILE="MINERvA_flux_ME1F.root"
FLUX_HIST="flux_E_cvweighted_CV_WithStatErr"
TARG="1000180400[1.00]"
NU_PDG=14
E_MIN=0.1
E_MAX=50.0
GEOM=argon_box_2m.gdml
NEVENTS=1000
EDEP_MAC=edep.mac

FLUX_NAME=NuMIME
GEN_NAME=GENIE10a
TUNE=G18_10a_00_000
TEMPLATE=batch_GENIEv3_${TUNE}_EDEPSIM_2D_TEMPLATE.sh

## Loop over jobs
for N in $(seq ${FIRST_JOB} ${LAST_JOB})
do
    printf -v PADJOB "%03d" ${N}

    OUTFILE_ROOT=nularbox_${FLUX_NAME}_${NU_PDG}_${GEN_NAME}_${PADJOB}

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
    
    echo "Submitting ${THIS_TEMP}"

    ## Submit the template
    sbatch ${THIS_TEMP}

    ## No need to delete, so done
    rm ${THIS_TEMP}
done
