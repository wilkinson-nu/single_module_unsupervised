# single_module_unsupervised

All of the dependencies are compiled into a container using the Dockerfile.MinkowskiEngine script. As a reference to myself, I do this with:
```
sudo docker build -f Dockerfile.MinkowskiEngine -t wilkinsonnu/ml_tools:ME --progress=plain --network=host .
```
And then push it to dockerhub, before pulling into shifter on NERSC

The container is available for any jobs submitting through slurm on Perlmutter (see `run_contrastive_jobs.sh` for a working example). To use a custom container as the kernel in JupyterLab, follow the instructions here: `https://docs.nersc.gov/services/jupyter/how-to-guides/`

To simply use this prebuilt docker image for your own jupyter notebook, on NERSC you can run
```

shifter --image=docker:wilkinsonnu/ml_tools:ME python -m ipykernel install --prefix $HOME/.local --name ml_env --display-name MinkowskiEngine
```
If successful, you'll get a message like `Installed kernelspec ml_env in {directory}`. In that directory, find the file called `kernel.json`. Add the two lines
```
"shifter",
"--image=docker:wilkinsonnu/ml_tools:ME",
```
to the beginning of the argv block. When done, the json file look something like
```
{
  "argv": [
  "shifter",
  "--image=docker:wilkinsonnu/ml_tools:ME",
  "/opt/conda/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "MinkowskiEngine",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
```
Now when you start a notebook in JupyterLab, you should be able to choose MinkowskiEngine from the list of available kernels.

# Preparing inputs
The inputs to this work are post "flow" output, the latest at time of writing are v9, found here: 
```
/global/cfs/cdirs/dune/www/data/Module1/TPC12/reflow-test/flowed_v9/charge_only/
```
They are processed to 2D sparse tensors using `make_2D_module_images_sparse_hdf5.py`, and `process_inputs.sh` is a convenience bash script for submitting a series of parallel jobs to process all of the input files. A simple script `simple_dataset_study.py` makes some high level plots using those processed images.

Processed images can be found in, which should be accessible to anybody with access to the DUNE allocation:
```
/pscratch/sd/c/cwilk/h5_inputs_v9
```

# Data augmentations
A critical choice for contrastive learning tasks is choosing appropriate augmentations to apply to the data. An example jupyter notebook for looking at the impact of augmentations (by default using the nominal set used for training), can be found in: `single_module_augmentation_tests_ME.ipynb`.

# Training the encoder
There's an example training jupyter notebook: `single_module_contrastive_training_ME.ipynb`, which is useful for testing new loss functions and looking at training dynamics. It can also optionally use a pre-trained model for a warm start for more advanced training studies.

To train at scale, `single_module_contrastive_dist_ME.py` allows for distributed training using multiple GPUs, and in principle multiple nodes (although currently that hasn't been tested). There are many arguments to the training script, and `run_contrastive_jobs.sh` provides an example bash script for submitting jobs with different combinations of those parameters.

# Evaluating the trained model
A messy jupyter notebook, which encodes a series of images, and then has a number of functions for investigating the distributions of images in encoded space can be found here: `single_module_pretrained_contrastive_ME.ipynb`. It includes example t-SNE, k nearest neighbour and DBSCAN algorithms, using (RAPIDS) cuML to use GPU-enabled implementations. Using the GPU, they're all pretty fast tested up to 500k images. There are also a number of plotting functions for convenience.
