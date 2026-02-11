#!/bin/bash
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --time=120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB

## These can change for each job
SEED=__SEED__
OUTDIR_ROOT=__OUTDIR_ROOT__
OUTFILE_ROOT=__OUTFILE_ROOT__
FLUX_FILE=__FLUX_FILE__
FLUX_HIST=__FLUX_HIST__
TARG=__TARG__
NU_PDG=__NU_PDG__
E_MIN=__E_MIN__
E_MAX=__E_MAX__
GEOM=__GEOM__
NEVENTS=__NEVENTS__
EDEP_MAC=__EDEP_MAC__

## Image making
IMAGE_SIZE=__IMAGE_SIZE__
MIN_HITS=__MIN_HITS__
THRESHOLD=__THRESHOLD__

## Fixed
INPUTS_DIR=${PWD}/MC_inputs
GENIE_TUNE=G18_10c_00_000

## Where to do stuff
tempDir=${SCRATCH}/${OUTFILE_ROOT}_${SEED}
echo "Moving to SCRATCH: ${tempDir}"
mkdir ${tempDir}
cd ${tempDir}

## Get the necessary inputs
cp ${INPUTS_DIR}/${GENIE_TUNE}_splines.xml.gz .
cp ${INPUTS_DIR}/${FLUX_FILE} .
cp ${INPUTS_DIR}/${EDEP_MAC} .
cp ${INPUTS_DIR}/${GEOM} .

## Get a hacked PDG table to include O11 (GENIE issue #305 on their github)
cp ${INPUTS_DIR}/mod_genie_pdg_table.txt .

## This is... pretty bad practice. Copy the run script and any library functions in the directory...
cp ${INPUTS_DIR}/../*.py .

## Step one, run the GENIE+INCL events in a different container
## This seems bizarre, but, the INCL code isn't generally available, and requires EL7 libraries...
echo "Starting gevgen..."
shifter --entrypoint --module=cvmfs --image=docker:wilkinsonnu/nuisance_project:genie_v3.2.0 /bin/bash -c \
	"source /cvmfs/fermilab.opensciencegrid.org/products/genie/bootstrap_genie_ups.sh; \
    	setup genie v3_02_00c -q e20:inclxx:prof; \
	export INCL_SRC_DIR=/cvmfs/fermilab.opensciencegrid.org/products/genie/local/inclxx/v5_2_9_5a/source; \
	export GENIE_PDG_TABLE=mod_genie_pdg_table.txt; \
	gevgen -n ${NEVENTS} -t ${TARG} -p ${NU_PDG} \
        --cross-sections ${GENIE_TUNE}_splines.xml.gz \
        --tune ${GENIE_TUNE} --seed ${SEED} \
        -f ${FLUX_FILE},${FLUX_HIST} -e ${E_MIN},${E_MAX} -o ${OUTFILE_ROOT}_GHEP.root &> /dev/null"

echo "Converting to rootracker..."
shifter --image=docker:wilkinsonnu/simple_det_sim:latest gntpc -i ${OUTFILE_ROOT}_GHEP.root -f rootracker -o ${OUTFILE_ROOT}_GROO.root

## Copy back the GENIE output file
if [ ! -d "${OUTDIR_ROOT}/GENIE" ]; then
    mkdir -p ${OUTDIR_ROOT}/GENIE
fi
cp ${tempDir}/${OUTFILE_ROOT}_GROO.root ${OUTDIR_ROOT}/GENIE/.

## Prepare the mac file
sed -i "s/_GEOM_/${GEOM}/g" ${EDEP_MAC}
sed -i "s/_GROO_FILE_/${OUTFILE_ROOT}_GROO.root/g" ${EDEP_MAC}
sed -i "s/_RAND1_/$((SEED + 1))/g" ${EDEP_MAC}
sed -i "s/_RAND2_/$((SEED + 2))/g" ${EDEP_MAC}

echo "Running edep-sim..."
shifter --image=docker:wilkinsonnu/simple_det_sim:latest edep-sim -o ${OUTFILE_ROOT}_EDEPSIM.root \
	${EDEP_MAC} \
	-e ${NEVENTS} &> /dev/null

## Copy back the edep-sim file
if [ ! -d "${OUTDIR_ROOT}/EDEPSIM" ]; then
    mkdir -p ${OUTDIR_ROOT}/EDEPSIM
fi
cp ${tempDir}/${OUTFILE_ROOT}_EDEPSIM.root ${OUTDIR_ROOT}/EDEPSIM/.

echo "Prepare 2D images..."
shifter --image=docker:wilkinsonnu/simple_det_sim:latest python3 make_2D_nusim_images.py --input ${OUTFILE_ROOT}_EDEPSIM.root --output ${OUTFILE_ROOT}_IMAGES2D.h5 \
	--image_size ${IMAGE_SIZE} --min_hits ${MIN_HITS} --threshold ${THRESHOLD}

## Copy back the images
if [ ! -d "${OUTDIR_ROOT}/IMAGES2D" ]; then
    mkdir -p ${OUTDIR_ROOT}/IMAGES2D
fi
cp ${tempDir}/${OUTFILE_ROOT}_IMAGES2D.h5 ${OUTDIR_ROOT}/IMAGES2D/.

## Clean up
rm -r ${tempDir}
