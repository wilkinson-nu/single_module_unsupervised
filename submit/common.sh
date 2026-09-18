## Threading (very important to avoid massive contention)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

## Python behaviour
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

## Essential for reducing ME VRAM usage
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

## NCCL on Perlmutter
export NCCL_SOCKET_IFNAME=hsn
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 20000))

## This function is for making a copy of the current repo state into the run directory
snapshot_repo() {
    local RUN_DIR=$1
    local REPO=$2
    : "${IMAGE:?The $IMAGE needs to be specified}"

    mkdir -p "$RUN_DIR"
    echo "Copying ${REPO} into ${RUN_DIR}"

    ## Copy the current repo state into the RUN_DIR
    rsync -a --exclude='.git' --exclude='__pycache__' "$REPO/src/" "$RUN_DIR/src/"
    rsync -a "$REPO/configs/" "$RUN_DIR/configs/"

    ## Some documentation to get exactly the code state
    git -C "$REPO" rev-parse HEAD        > "$RUN_DIR/git-sha.txt"
    git -C "$REPO" status --porcelain    > "$RUN_DIR/git-dirty.txt"
    git -C "$REPO" diff                  > "$RUN_DIR/git-diff.patch"
    cp "${BASH_SOURCE[-1]}" "$RUN_DIR/submit.sh"

    ## Compile the package into the RUN_DIR
    shifter --image=$IMAGE python3 -m compileall -q "$RUN_DIR/src"
    export PYTHONPATH="$RUN_DIR/src"
}
