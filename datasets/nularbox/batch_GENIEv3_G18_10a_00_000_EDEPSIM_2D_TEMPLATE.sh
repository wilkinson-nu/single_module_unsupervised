#!/bin/bash
#SBATCH --image=docker:wilkinsonnu/simple_det_sim:latest
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

## Fixed
INPUTS_DIR=${PWD}/MC_inputs
GENIE_TUNE=G18_10a_00_000

## Where to do stuff
tempDir=${SCRATCH}/${OUTFILE/.root/}_${SEED}
echo "Moving to SCRATCH: ${tempDir}"
mkdir ${tempDir}
cd ${tempDir}

## Get the necessary inputs
cp ${INPUTS_DIR}/${TUNE}_splines.xml.gx .
cp ${INPUTS_DIR}/${FLUX_FILE} .
cp ${INPUTS_DIR}/${EDEP_MAC} .
cp ${INPUTS_DIR}/${GEOM} .
cp ${INPUTS_DIR}/../make_2D_nusim_images.py .

echo "Starting gevgen..."
shifter --entrypoint gevgen -n ${NEVENTS} -t ${TARG} -p ${NU_PDG} \
        --cross-sections ${GENIE_TUNE}_splines.xml.gz \
        --tune ${TUNE} --seed ${SEED} \
        -f ${FLUX_FILE},${FLUX_HIST} -e ${E_MIN},${E_MAX} -o ${OUTFILE_ROOT}_GHEP.root

echo "Converting to rootracker..."
shifter --entrypoint gntpc -i ${OUTFILE_ROOT}_GHEP.root -f rootracker -o ${OUTFILE_ROOT}_GROO.root

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
shifter --entrypoint edep-sim -o ${OUTFILE_ROOT}_EDEPSIM.root \
	${EDEP_MAC} \
	-e ${NEVENTS}

## Copy back the edep-sim file
if [ ! -d "${OUTDIR_ROOT}/EDEPSIM" ]; then
    mkdir -p ${OUTDIR_ROOT}/EDEPSIM
fi
cp ${tempDir}/${OUTFILE_ROOT}_EDEPSIM.root ${OUTDIR_ROOT}/EDEPSIM/.

echo "Prepare 2D images..."
shifter --entrypoint python3 make_2D_nusim_images.py ${OUTFILE_ROOT}_EDEPSIM.root ${OUTFILE_ROOT}_IMAGES2D.h5

## Copy back the images
if [ ! -d "${OUTDIR_ROOT}/IMAGES2D" ]; then
    mkdir -p ${OUTDIR_ROOT}/IMAGES2D
fi
cp ${tempDir}/${OUTFILE_ROOT}_IMAGES2D.root ${OUTDIR_ROOT}/IMAGES2D/.

## Clean up
rm -r ${tempDir}
