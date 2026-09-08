#!/bin/bash
#SBATCH --image=docker:wilkinsonnu/simple_det_sim:latest
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --time=360
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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
EXIT_DOWNSTREAM=__EXIT_DOWNSTREAM__

## Fixed
INPUTS_DIR=${PWD}/MC_inputs
GENIE_TUNE=G18_10a_00_000

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

## Sort out the decay behaviour
mkdir xml_override
cp ${INPUTS_DIR}/CommonDecay.xml xml_override/.

## This is... pretty bad practice. Copy the run script and any library functions in the directory...
cp ${INPUTS_DIR}/../*.py .
cp ${INPUTS_DIR}/../../*.py .

echo "Starting gevgen..."
shifter gevgen -n ${NEVENTS} -t ${TARG} -p ${NU_PDG} \
	--event-generator-list CC \
        --cross-sections ${GENIE_TUNE}_splines.xml.gz \
	--xml-path xml_override \
	--tune ${GENIE_TUNE} --seed ${SEED} \
        -f ${FLUX_FILE},${FLUX_HIST} -e ${E_MIN},${E_MAX} -o ${OUTFILE_ROOT}_GHEP.root

echo "Converting to rootracker..."
shifter gntpc -i ${OUTFILE_ROOT}_GHEP.root -f rootracker -o ${OUTFILE_ROOT}_GROO.root

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
shifter edep-sim -o ${OUTFILE_ROOT}_EDEPSIM.root \
	${EDEP_MAC} \
	-e ${NEVENTS} &> /dev/null

## Copy back the edep-sim file
## if [ ! -d "${OUTDIR_ROOT}/EDEPSIM" ]; then
##     mkdir -p ${OUTDIR_ROOT}/EDEPSIM
## fi
## cp ${tempDir}/${OUTFILE_ROOT}_EDEPSIM.root ${OUTDIR_ROOT}/EDEPSIM/.

echo "Prepare images..."
## Make 2 sets:
## - One where containment is required in the full image
## - Another where containment is in a smaller region
shifter python3 make_2D_nusim_images.py \
	--input ${OUTFILE_ROOT}_EDEPSIM.root \
	--output ${OUTFILE_ROOT}_IMAGES_CCCONT512.h5 \
	--image_size ${IMAGE_SIZE} \
	--offset 0 0 -128 \
	--box_size ${IMAGE_SIZE} \
	--exit_downstream ${EXIT_DOWNSTREAM} \
	--min_hits ${MIN_HITS} \
	--threshold ${THRESHOLD}

## Copy back the images
if [ ! -d "${OUTDIR_ROOT}/IMAGES_CCCONT512" ]; then
    mkdir -p ${OUTDIR_ROOT}/IMAGES_CCCONT512
fi
cp ${tempDir}/${OUTFILE_ROOT}_IMAGES_CCCONT512.h5 ${OUTDIR_ROOT}/IMAGES_CCCONT512/.

shifter python3 make_2D_nusim_images.py \
	--input ${OUTFILE_ROOT}_EDEPSIM.root \
	--output ${OUTFILE_ROOT}_IMAGES_CCCONT256.h5 \
	--image_size ${IMAGE_SIZE} \
        --offset 0 0 -128 \
	--box_size 256 \
	--box_offset 0 0 -64 \
	--exit_downstream ${EXIT_DOWNSTREAM} \
	--min_hits ${MIN_HITS} \
        --threshold ${THRESHOLD}

## Copy back the images
if [ ! -d "${OUTDIR_ROOT}/IMAGES_CCCONT256" ]; then
    mkdir -p ${OUTDIR_ROOT}/IMAGES_CCCONT256
fi
cp ${tempDir}/${OUTFILE_ROOT}_IMAGES_CCCONT256.h5 ${OUTDIR_ROOT}/IMAGES_CCCONT256/.

## Clean up
rm -r ${tempDir}
