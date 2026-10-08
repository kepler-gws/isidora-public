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
    parser.add_argument('input_tile_pickle_path', type=pathlib.Path)
    parser.add_argument('input_rb_cache_path', type=pathlib.Path)
    parser.add_argument('output_dir', type=pathlib.Path)
    parser.add_argument('--input-tile-training-wf-arr-path', type=pathlib.Path, default=None)
    parser.add_argument('--n-output-tiles', type=int, required=True)
    parser.add_argument('--split-strategy', type=str, choices=['minMc_freq', 'n_training_wf'], required=True)
    parser.add_argument('--n_freq_overlap_oneside', type=int, default=2)
    args = parser.parse_args()
    
    with open(args.input_tile_pickle_path, 'rb') as f:
        input_tile = pickle.load(f)
    with open(args.input_rb_cache_path, 'rb') as f:
        input_rb = pickle.load(f)
    if args.input_tile_training_wf_arr_path:
        print(f"Loading training waveforms from {args.input_tile_training_wf_arr_path}")
        cached_training_wf_arr = np.lib.format.open_memmap(args.input_tile_training_wf_arr_path, mode='r')
        # cached_training_wf_arr = np.load(args.input_tile_training_wf_arr_path)
    else:
        cached_training_wf_arr = None
        
    input_roq_size = isidora.training.n_roq_weights(input_rb.basis.Nbasis_)
    
    subtile_dict, subtile_param_ind_dict = isidora.training.split_tile(input_tile,
                                                                       args.n_output_tiles,
                                                                       strategy=args.split_strategy,
                                                                       n_freq_overlap_oneside=args.n_freq_overlap_oneside,)
    
    subtile_rb_dict = {}    
    reconstructed_wf_dict = {}
    roq_size_dict = {}
    for subtile_freq_lim, subtile in subtile_dict.items():
        start_time = time.time()
        print(f"Processing subtile with frequency limits {subtile_freq_lim}")
        subtitle_training_wf_arr = np.array(cached_training_wf_arr[subtile_param_ind_dict[subtile_freq_lim]]) if cached_training_wf_arr is not None else None
        subtile_rb = isidora.training.tile_to_reduced_basis(subtile, skip_het=True, greedy_tol=1e-12,
                                                            cache_path=None,
                                                            cached_training_wf_arr=subtitle_training_wf_arr)[0]
        subtile_rb_dict[subtile_freq_lim] = subtile_rb
        reconstructed_wf_dict[subtile_freq_lim] = isidora.training.rb_reconstructed_wf_factory(subtile, subtile_rb, skip_het=True)
        roq_size_dict[subtile_freq_lim] = isidora.training.n_roq_weights(subtile_rb.basis.Nbasis_)
        end_time = time.time()
        print(f"Generated RB on subtile with frequency limits {subtile_freq_lim} with {subtile_rb.basis.Nbasis_} elems in {end_time - start_time} seconds")
        
    print(f'Total ROQ size across all tiles: {sum(roq_size_dict.values())}, compared to input tile ROQ size of {input_roq_size}')
    print(f'Compression of {sum(roq_size_dict.values()) / input_roq_size:.3f} compared to input tile')
    
    roq_size_dict = roq_size_dict | {'strategy': args.split_strategy, 
                     'n_output_tiles': args.n_output_tiles,
                     'n_freq_overlap_oneside': args.n_freq_overlap_oneside}
    
    with open(args.output_dir / 'subtile_splits.pkl', 'wb') as f:
        pickle.dump((subtile_dict, subtile_param_ind_dict), f)
    with open(args.output_dir / 'subtile_rb_dict.pkl', 'wb') as f:
        pickle.dump(subtile_rb_dict, f)
    with open(args.output_dir / 'reconstructed_wf_dict.pkl', 'wb') as f:
        pickle.dump(reconstructed_wf_dict, f)
    with open(args.output_dir / 'roq_size_dict.pkl', 'wb') as f:
        pickle.dump(roq_size_dict, f)
