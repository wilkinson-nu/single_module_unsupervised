# single_module_unsupervised

All of the dependencies are compiled into a container using the Dockerfile.MinkowskiEngine script. As a reference to myself, I do this with:
```
sudo docker build -f Dockerfile.MinkowskiEngine -t wilkinsonnu/ml_tools:ME --progress=plain --network=host .
```
And then push it to dockerhub, before pulling into shifter on NERSC

The container is available for any jobs submitting through slurm on Perlmutter (see `run_contrastive_jobs.sh` for a working example). To use a custom container as the kernel in JupyterLab, follow the instructions here: `https://docs.nersc.gov/services/jupyter/how-to-guides/`

# Preparing inputs
The inputs to this work are post "flow" output, the latest at time of writing are v9, found here: 
```
/global/cfs/cdirs/dune/www/data/Module1/TPC12/reflow-test/flowed_v9/charge_only/
```
They are processed to 2D sparse tensors using `make_2D_module_images_sparse_hdf5.py`, and `process_inputs.sh` is a convenience bash script for submitting a series of parallel jobs to process all of the input files. A simple script `simple_dataset_study.py` makes some high level plots using those processed images.

# Training the encoder
There's an example training jupyter notebook: `single_module_contrastive_training_ME.ipynb`, which is useful for testing new loss functions and looking at training dynamics. It can also optionally use a pre-trained model for a warm start for more advanced training studies.

To train at scale, `single_module_contrastive_dist_ME.py` allows for distributed training using multiple GPUs, and in principle multiple nodes (although currently that hasn't been tested). There are many arguments to the training script, and `run_contrastive_jobs.sh` provides an example bash script for submitting jobs with different combinations of those parameters.
