#!/bin/bash

export OMP_NUM_THREADS=16
export MASTER_ADDR=$(hostname)
export MASTER_PORT='12355'
export WORLD_SIZE=4
