import os
import sys
import copy
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
from sklearn.model_selection import KFold
import tqdm

import isidora.training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_tile_pickle_path', type=pathlib.Path)
    parser.add_argument('input_rb_cache_path', type=pathlib.Path)
    parser.add_argument('validation_tile_pickle_path', type=pathlib.Path)
    parser.add_argument('output_dir', type=pathlib.Path)
    parser.add_argument('--input-tile-training-wf-arr-path', type=pathlib.Path, required=True)
    parser.add_argument('--input-tile-validation-wf-arr-path', type=pathlib.Path, required=True)
    parser.add_argument('--subtile-result-path', type=pathlib.Path, required=True)
    parser.add_argument('--subtile-rb-path', type=pathlib.Path, required=True)
    parser.add_argument('--validation-wf-norm', type=str, choices=['L2', 'Linf'], default='Linf')
    parser.add_argument('--validate-full-rb', action='store_true', help='If set, will validate the full RB as well')
    args = parser.parse_args()
    
    with open(args.input_tile_pickle_path, 'rb') as f:
        input_tile = pickle.load(f)
    with open(args.input_rb_cache_path, 'rb') as f:
        input_rb = pickle.load(f)
    with open(args.validation_tile_pickle_path, 'rb') as f:
        validation_tile = pickle.load(f)

    cached_training_wf_arr = np.lib.format.open_memmap(args.input_tile_training_wf_arr_path, mode='r')
    cached_validation_wf_arr = np.lib.format.open_memmap(args.input_tile_validation_wf_arr_path, mode='r')
    
    if args.validation_wf_norm == 'L2':
        wf_norm_func = lambda wf: np.sqrt(np.sum(wf**2))
    elif args.validation_wf_norm == 'Linf':
        wf_norm_func = lambda wf: np.max(np.abs(wf))
        
    with open(args.subtile_result_path, 'rb') as f:
        subtile_dict, subtile_param_ind_dict = pickle.load(f)
        
    with open(args.subtile_rb_path, 'rb') as f:
        subtile_rb_dict = pickle.load(f)
        
    if args.validate_full_rb:
        full_rb_reconstructed_wf_func = isidora.training.rb_reconstructed_wf_factory(input_tile, input_rb, skip_het=True)
        
        def test_full_rb_error(test_ind):
            test_param = validation_tile.tile_full_gw_param_list[test_ind]
            test_wf = cached_validation_wf_arr[test_ind]
            reconstructed_test_wf = full_rb_reconstructed_wf_func(validation_tile.physical_time, test_param)
            return wf_norm_func(test_wf - reconstructed_test_wf)

        full_rb_validation_errors = {i: test_full_rb_error(i) for i in tqdm.tqdm(range(len(cached_validation_wf_arr)))}
        print(f'Full RB validation error avg | max = {np.mean(list(full_rb_validation_errors.values())):.3e} | {max(full_rb_validation_errors.values()):.3e}')
        with open(args.output_dir / 'full_rb_validation_results.pkl', 'wb') as f:
            pickle.dump(full_rb_validation_errors, f)
        
    validation_results = {'wf_norm': args.validation_wf_norm}
        
    for subtile_freq_lim, subtile in subtile_dict.items():
        subtile_reconstructed_wf_func = isidora.training.rb_reconstructed_wf_factory(subtile, subtile_rb_dict[subtile_freq_lim], skip_het=True)
        
        validation_param_freqs = u.Quantity([p['initial_freq'] for p in validation_tile.tile_full_gw_param_list], unit=u.Hz)
        validation_param_inds = np.argwhere((validation_param_freqs >= subtile_freq_lim[0]) & (validation_param_freqs <= subtile_freq_lim[1]))        
            
        def test_error(test_ind):
            test_param = validation_tile.tile_full_gw_param_list[test_ind]
            test_wf = cached_validation_wf_arr[test_ind]
            reconstructed_test_wf = subtile_reconstructed_wf_func(validation_tile.physical_time, test_param)
            return wf_norm_func(test_wf - reconstructed_test_wf)

        validation_results[subtile_freq_lim] = {i: test_error(i) for i in tqdm.tqdm(validation_param_inds.flatten())}
        print(f'Frequency limits {subtile_freq_lim}: validation {args.validation_wf_norm} error avg | max = {np.mean(list(validation_results[subtile_freq_lim].values())):.3e} | {max(validation_results[subtile_freq_lim].values()):.3e}')

    with open(args.output_dir / 'validation_results.pkl', 'wb') as f:
        pickle.dump(validation_results, f)