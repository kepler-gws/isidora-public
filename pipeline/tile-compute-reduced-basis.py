import os
import sys
from dotenv import find_dotenv
sys.path.append(os.path.dirname(find_dotenv()))

import argparse
import pathlib
import time

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import dill as pickle
import isidora.training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tile_pickle_path', type=pathlib.Path)
    parser.add_argument('rb_cache_path', type=pathlib.Path)
    parser.add_argument('--tile-training-wf-arr-path', type=pathlib.Path, default=None)
    args = parser.parse_args()
    
    start_time = time.time()
    with open(args.tile_pickle_path, 'rb') as f:
        tile = pickle.load(f)
    if args.tile_training_wf_arr_path:
        print(f"Loading training waveforms from {args.tile_training_wf_arr_path}")
        # cached_training_wf_arr = np.lib.format.open_memmap(args.tile_training_wf_arr_path, mode='r')
        cached_training_wf_arr = np.load(args.tile_training_wf_arr_path)
    else:
        cached_training_wf_arr = None
    reduced_basis, _, cached_training_wf_arr = isidora.training.tile_to_reduced_basis(tile, skip_het=True, greedy_tol=1e-12,
                                                                                      cache_path=args.rb_cache_path, refresh_cache=True,
                                                                                      cached_training_wf_arr=cached_training_wf_arr)
    end_time = time.time()
    print(f"Generated RB on tile with frequency limits {tile.tile_freq_limits} and {len(tile.tile_full_gw_param_list)} training waveforms in {end_time - start_time} seconds")