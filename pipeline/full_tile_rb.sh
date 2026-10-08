#!/bin/bash
conda activate isidora
python3 tile-compute-reduced-basis.py data/training_set_full_tile.pkl data/full_tile_rb.pkl --tile-training-wf-arr-path data/training_set_full_wf.npy
