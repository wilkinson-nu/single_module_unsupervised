#!/bin/bash

## Control the number of jobs to spawn
FIRST_JOB=0
LAST_JOB=9

## SBND (SciBooNE) flux
EXPT_NAME="SBND"
FLUX_FILE="SciBooNE_numu_flux.root"
FLUX_HIST="numu"
E_MIN=0.1
E_MAX=5.0

## DUNE FLUX
EXPT_NAME="DUNEND"
FLUX_FILE="DUNE_OptimizedEngineeredNov2017_REGULAR.root"
FLUX_HIST="numu_NDFHC_flux"
E_MIN=0.1
E_MAX=50.0

## NuMI ME flux
EXPT_NAME="NuMIME"
FLUX_FILE="MINERvA_flux_ME1F.root"
FLUX_HIST="flux_E_cvweighted_CV_WithStatErr"
E_MIN=0.1
E_MAX=50.0

## Common configuration
TARG="1000180400[1.00]"
NU_PDG=14
GEOM=argon_box_3m.gdml
NEVENTS=25000
EDEP_MAC=edep.mac

## Pick the GENIE version
GEN_VERSION=10a
GEN_NAME=GENIE${GEN_VERSION}
TEMPLATE=batch_${GEN_NAME}_EDEPSIM_TEMPLATE.sh

## Image making options
IMAGE_SIZE=512
EXIT_DOWNSTREAM=1
MIN_HITS=10
THRESHOLD=0

## Directory to save to
OUTNAME=${GEN_NAME}_${TUNE}_${EXPT_NAME}
OUTDIR_ROOT="${CFS}/users/${USER}/NULARBOX/${OUTNAME}"

## Get the REPO path based on where we are
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

## Use software here (e.g., not in home)
SOFTWARE_DIR=$PSCRATCH/shared_larch

## Decide where the software lives, and whether we need to populate it
if [[ -z "${SOFTWARE_DIR:-}" ]]; then
    SOFTWARE_DIR=$PSCRATCH/shared_larch_$(date +%Y%m%d_%H%M%S)
fi

if [[ -d "$SOFTWARE_DIR/src" ]]; then
    echo "Reusing software copy: ${SOFTWARE_DIR}"
else
    echo "Creating software copy: ${SOFTWARE_DIR}"
    mkdir -p "$SOFTWARE_DIR"

    rsync -a --exclude='.git' --exclude='__pycache__' "$REPO/src/" "$SOFTWARE_DIR/src/"
    cp "$REPO/generation/make_nusim_images.py" "$SOFTWARE_DIR/"

    git -C "$REPO" rev-parse HEAD     > "$SOFTWARE_DIR/git-sha.txt"
    git -C "$REPO" status --porcelain > "$SOFTWARE_DIR/git-dirty.txt"
    git -C "$REPO" diff              > "$SOFTWARE_DIR/git-diff.patch"
    cp "${BASH_SOURCE[0]}"             "$SOFTWARE_DIR/spawner.sh"

    shifter --image=docker:wilkinsonnu/simple_det_sim:latest python3 -m compileall -q "$SOFTWARE_DIR/src"
fi

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
    sed -i "s/__SOFTWARE_DIR__/${SOFTWARE_DIR}/g" ${THIS_TEMP}
    
    echo "Submitting ${THIS_TEMP}"

    ## Submit the template
    sbatch ${THIS_TEMP}

    ## No need to delete, so done
    rm ${THIS_TEMP}
done
