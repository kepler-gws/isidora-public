import os
import sys
from dotenv import find_dotenv
sys.path.append(os.path.dirname(find_dotenv()))

import itertools
from dataclasses import dataclass, field
import copy

import numpy as np
import astropy.units as u
import astropy.constants as const
import dill as pickle
import matplotlib.pyplot as plt
import arby
import arby.basis
import tqdm

import isidora.utils


@dataclass
class HeterodyneFreqTile:
    tile_freq_limits: tuple
    tile_ref_gw_params: dict = None
    tile_ref_gw_cycles: float = 0
    tile_full_gw_param_list: list = field(default_factory=list)
    physical_time: np.ndarray = None
    amp_func: callable = None
    phase_func: callable = None
    freq_func: callable = None
    prev_tile: list['HeterodyneFreqTile'] = None
    next_tile: list['HeterodyneFreqTile'] = None
    
    def generate_full_waveform(self, param, skip_het=True):
        param = param.copy()
        if 'init_phi' not in param:
            param['init_phi'] = 0
        unhet_wf = self.amp_func(self.physical_time, param) * np.exp(1j * self.phase_func(self.physical_time, param))
        if skip_het or self.tile_ref_gw_params is None:
            return unhet_wf
        else:
            ref_wf = self.amp_func(self.physical_time, self.tile_ref_gw_params) * np.exp(1j * self.phase_func(self.physical_time, self.tile_ref_gw_params))
            return unhet_wf / ref_wf
        
def tile_to_reduced_basis(tile, init_phase=[0], skip_het=False, greedy_tol=1e-12,
                          cached_training_wf_arr=None,
                          cache_path=None, refresh_cache=False):
    amp_func, phase_func = tile.amp_func, tile.phase_func
    t_ = tile.physical_time
    if skip_het or tile.tile_ref_gw_params is None:
        print('Warning: not heterodyning')
        reference_wf = 1
    else:
        reference_wf = amp_func(t_, tile.tile_ref_gw_params | {'init_phi': 0}) * np.exp(1j * phase_func(t_, tile.tile_ref_gw_params | {'init_phi': 0}))
    assert np.all(np.isfinite(reference_wf))
    if cache_path and os.path.exists(cache_path) and not refresh_cache:
        with open(cache_path, 'rb') as f:
            reduced_basis = pickle.load(f)
        return reduced_basis, reference_wf, cached_training_wf_arr
    else:
        if cached_training_wf_arr is None:
            cached_training_wf_arr = []
            for partial_param in tile.tile_full_gw_param_list:
                for ip in init_phase:
                    cand_wf = amp_func(t_, partial_param | {'init_phi': ip}) * np.exp(1j * phase_func(t_, partial_param | {'init_phi': ip}))
                    try:
                        assert np.all(np.isfinite(cand_wf))
                    except AssertionError:
                        print(f'Non-finite waveform for params {partial_param} and init phase {ip}')
                        plt.plot(t_, np.real(cand_wf), label='real')
                        raise
                    cached_training_wf_arr.append(cand_wf / reference_wf)
            cached_training_wf_arr = np.array(cached_training_wf_arr)
        else:
            print(f'Warning: using provided cached training waveforms, which may or not be consistent with skip_het parameter for this function.')
        print(f'Training set size is {len(cached_training_wf_arr)}')
        reduced_basis = arby.basis.reduced_basis(training_set=cached_training_wf_arr, 
                                                physical_points=t_.to(u.s).value,
                                                greedy_tol=greedy_tol,
                                                normalize=True)
        print(f'Reduced basis has {reduced_basis.basis.Nbasis_} elements.')
        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(reduced_basis, f)
        return reduced_basis, reference_wf, cached_training_wf_arr

def rb_reconstructed_wf_factory(tile, tile_rb, skip_het=False):
    exact_eval_func = lambda t_, param: tile.amp_func(t_, param) * np.exp(1j * tile.phase_func(t_, param))
    if skip_het or tile.tile_ref_gw_params is None:
        reference_wf = lambda t_: 1
    else:
        reference_wf = lambda t_: tile.amp_func(t_, tile.tile_ref_gw_params | {'init_phi': 0}) * np.exp(1j * tile.phase_func(t_, tile.tile_ref_gw_params | {'init_phi': 0}))
    exact_het_eval_func = lambda t_, param: exact_eval_func(t_, param) / reference_wf(t_)
    eim = tile_rb.basis.eim_
    def exact_eim_reconst(t_, param):
        h_at_nodes = exact_het_eval_func(t_[eim.nodes], param)
        eim_reconst = (eim.interpolant @ np.array(h_at_nodes)) * reference_wf(t_)
        return eim_reconst
    return exact_eim_reconst

def n_roq_weights(n_rb_elem):
    linear_roq_weights = n_rb_elem
    quadratic_roq_weights = 2 * n_rb_elem**2
    return linear_roq_weights + quadratic_roq_weights

def group_param_list(param_list, grouping_param):
    '''Group a list of parameter dictionaries by the value of a specific parameter. Returns a dictionary mapping parameter values to lists of parameter dictionaries.'''
    grouped_dict = {}
    grouped_dict_indices = {}
    for i, param in enumerate(param_list):
        key = param[grouping_param]
        if key not in grouped_dict:
            grouped_dict[key] = []
        if key not in grouped_dict_indices:
            grouped_dict_indices[key] = []
        grouped_dict[key].append(param)
        grouped_dict_indices[key].append(i)
    # Sort by key
    grouped_dict = dict(sorted(grouped_dict.items()))
    grouped_dict_indices = dict(sorted(grouped_dict_indices.items()))
    return grouped_dict, grouped_dict_indices

def split_tile(starting_tile: HeterodyneFreqTile, n_output_tiles: int, strategy='minMc_freq', n_freq_overlap_oneside=2):
    '''Take an input tile and split into a dictionary of smaller tiles, along the initial frequency parameter only.
    Returns a dictionary mapping frequency intervals [a, b) to tiles, as well as a dictionary with the same keys, where
    the values are lists of indices into the param_list of the original tile (so a training set of waveforms can also be split).

    Args:
        starting_tile (HeterodyneFreqTile): The input tile to split.
        n_output_tiles (int): The number of output tiles to create.
        strategy (str, optional): The strategy for splitting the tile. Defaults to 'minMc_freq'. Must be one of 'minMc_freq' or 'n_training_wf'.
         - 'minMc_freq': Divide the grid into equal intervals in initial frequency along the lowest-Mc edge.
         - 'n_training_wf': Divide the grid into intervals in initial frequency such that 
           each tile has approximately the same number of training waveforms. 'Columns' in Mc will stay together.
        n_freq_overlap_oneside (int, optional): The dividing frequency will always be halfway between grid points in initial_freq.
                                                For the tiles on each side, take n_freq_overlap_oneside columns extra beyond the dividing frequency.
    Returns:
        dict: A dictionary mapping frequency intervals [a, b) to tiles.
        dict: A dictionary mapping frequency intervals [a, b) to lists of indices into the original tile's param_list.
    '''
    assert strategy in ['minMc_freq', 'n_training_wf']
    
    if n_output_tiles == 1:
        return {starting_tile.tile_freq_limits: starting_tile}, {starting_tile.tile_freq_limits: list(range(len(starting_tile.tile_full_gw_param_list)))}
    
    starting_param_by_freq, starting_param_ind_by_freq = group_param_list(starting_tile.tile_full_gw_param_list, 'initial_freq')

    unique_freq_hz = list(starting_param_by_freq.keys())
    unique_freq_hz = np.array([fr.to(u.Hz).value for fr in unique_freq_hz])
    assert np.all(np.diff(unique_freq_hz) > 0)

    starting_freq_lim = (unique_freq_hz.min() * u.Hz, unique_freq_hz.max() * u.Hz)
    
    if strategy == 'minMc_freq':
        dividing_freqs_tmp = np.linspace(starting_freq_lim[0].to(u.Hz).value, starting_freq_lim[1].to(u.Hz).value, n_output_tiles + 1) * u.Hz
        # Guarantee dividing frequencies are halfway between grid points in initial_freq
        dividing_freqs = [starting_freq_lim[0]]
        for f in dividing_freqs_tmp[1:-1]:
            f = f.to(u.Hz).value
            idx_right = np.searchsorted(unique_freq_hz, f, side='right')
            idx_left = idx_right - 1
            dividing_freqs.append((unique_freq_hz[idx_left] + unique_freq_hz[idx_right]) / 2 * u.Hz)
        dividing_freqs.append(starting_freq_lim[1])
        print(f'Dividing log-freqs: {[np.log10(f.to(u.Hz).value) for f in dividing_freqs]}')
    elif strategy == 'n_training_wf':
        n_total_params = len(starting_tile.tile_full_gw_param_list)
        n_params_per_tile = np.ceil(n_total_params / n_output_tiles)
        params_in_bin = 0
        dividing_freqs = [starting_freq_lim[0]]
        n_params_per_bin = []
        for i, freq in enumerate(starting_param_by_freq.keys()):
            Mc_col = starting_param_by_freq[freq]
            params_in_bin += len(Mc_col)
            if params_in_bin >= n_params_per_tile:
                dividing_freq = (freq + list(starting_param_by_freq.keys())[i + 1]) / 2
                dividing_freqs.append(dividing_freq)
                n_params_per_bin.append(params_in_bin)
                params_in_bin = 0
        n_params_per_bin.append(params_in_bin)
        dividing_freqs.append(starting_freq_lim[1])
        print(f'Dividing log-freqs: {[np.log10(f.to(u.Hz).value) for f in dividing_freqs]} with n params per tile {n_params_per_bin}')

    assert len(dividing_freqs) == n_output_tiles + 1

    tile_dict = {}
    tile_param_idx_dict = {}
    for i in range(n_output_tiles):
        lower_f, upper_f = dividing_freqs[i], dividing_freqs[i + 1]
        lower_f_col_ind = np.searchsorted(unique_freq_hz, lower_f.to(u.Hz).value, side='right')
        upper_f_col_ind = np.searchsorted(unique_freq_hz, upper_f.to(u.Hz).value, side='left')
        lower_f_col_ind = max(0, lower_f_col_ind - n_freq_overlap_oneside)
        upper_f_col_ind = min(len(starting_param_by_freq), upper_f_col_ind + n_freq_overlap_oneside)
        param_indices = sum(list(starting_param_ind_by_freq.values())[lower_f_col_ind:upper_f_col_ind], [])
        assert set(np.unique(param_indices)) == set(param_indices)
        new_tile = HeterodyneFreqTile(tile_freq_limits=(lower_f, upper_f),
                                      tile_full_gw_param_list=[starting_tile.tile_full_gw_param_list[i] for i in param_indices],
                                      physical_time=starting_tile.physical_time,
                                      amp_func=starting_tile.amp_func,
                                      phase_func=starting_tile.phase_func,
                                      freq_func=starting_tile.freq_func)
        tile_dict[(lower_f, upper_f)] = new_tile
        tile_param_idx_dict[(lower_f, upper_f)] = param_indices
    return tile_dict, tile_param_idx_dict

def freq_Mc_grid(t_, base_param_dict, gen_func,
                 freq_limits, Mc_limits,
                 n_freq_grid=300, n_Mc_grid=301,
                 freq_logspacing=False,
                 Mc_logspacing=True,
                 reject_nonfinite=True,
                 skip_gen_errors=False,
                 progress_bar=False,):
    '''Return a grid in frequency/chirp mass for some generating function and a base set of parameters.

    Args:
        t_ (ndarray): Unitful array of observing times; gets passed into gen_func.
        base_param_dict (dict): Dictionary of base parameters.
        gen_func (callable): Function to generate waveforms. Must have signature gen_func(t_, params).
        freq_limits (tuple): Unitful tuple of frequency limits.
        Mc_limits (tuple): Unitful tuple of chirp mass limits.
        n_freq_grid (int, optional): Number of frequency points. Defaults to 300.
        n_Mc_grid (int, optional): Number of chirp mass points. Defaults to 301.
        freq_logspacing (bool, optional): Whether to use log10 spacing for frequency. Defaults to False.
        Mc_logspacing (bool, optional): Whether to use log10 spacing for chirp mass. Defaults to True.
        reject_nonfinite (bool, optional): Whether to skip outputs that have any non-finite values. Defaults to True.
    '''
    if freq_logspacing:
        init_freq = np.logspace(np.log10(freq_limits[0].to(u.Hz).value), np.log10(freq_limits[1].to(u.Hz).value), n_freq_grid) * u.Hz
    else:
        init_freq = np.linspace(freq_limits[0].to(u.Hz).value, freq_limits[1].to(u.Hz).value, n_freq_grid) * u.Hz

    if Mc_logspacing:
        chirp_mass = np.logspace(np.log10(Mc_limits[0].to(u.Msun).value), np.log10(Mc_limits[1].to(u.Msun).value), n_Mc_grid) * u.Msun
    else:
        chirp_mass = np.linspace(Mc_limits[0].to(u.Msun).value, Mc_limits[1].to(u.Msun).value, n_Mc_grid) * u.Msun
        
    param_list = []
    outputs = []
    iterable = itertools.product(init_freq, chirp_mass)
    if progress_bar:
        iterable = tqdm.tqdm(iterable, total=n_freq_grid*n_Mc_grid)
    for i_f, Mc in iterable:
        params = base_param_dict.copy()
        params['initial_freq'] = i_f
        params['Mc'] = Mc
        try:
            out = gen_func(t_, params)
        except Exception:
            if skip_gen_errors:
                outputs.append(np.full_like(out, np.nan))
                param_list.append(params)
                continue
            else:
                print(f"Error generating output in grid occurred with params {params}:")
                raise
        if reject_nonfinite and np.any(~np.isfinite(out)):
            outputs.append(np.full_like(out, np.nan))
        else:
            outputs.append(gen_func(t_, params))
        param_list.append(params)
    return (init_freq, chirp_mass), param_list, outputs

def freq_Mc_grid_dynamic(t_, base_param_dict, gen_func, phase_func,
                         freq_limits, Mc_limits,
                         pts_per_cycle_freq=0.5, pts_per_cycle_Mc=0.5,
                         freq_logspacing=False,
                         Mc_logspacing=False,
                         reject_nonfinite=True,
                         skip_gen_errors=True,
                         progress_bar=False,
                         verbose=False):
    '''Return a dynamically spaced in frequency/chirp mass for some generating function and a base set of parameters.

    Args:
        t_ (ndarray): Unitful array of observing times; gets passed into phase_func.
        base_param_dict (dict): Dictionary of base parameters.
        gen_func (callable): Function to generate waveforms. Must have signature gen_func(t_, params).
        phase_func (callable): Function to generate GW phase. Must have signature phase_func(t_, params).
        init_freq_limits (tuple): Unitful tuple of initial frequency limits.
        init_Mc_limits (tuple): Unitful tuple of initial chirp mass limits.
        pts_per_cycle_freq (int, optional): Number of frequency points per GW cycle (for lowest chirp mass). Defaults to 2.
        pts_per_cycle_Mc (int, optional): Number of chirp mass points per GW cycle, for a specific frequency. Defaults to 1.
        freq_logspacing (bool, optional): Whether to use log10 spacing for frequency. Defaults to False.
        Mc_logspacing (bool, optional): Whether to use log10 spacing for chirp mass. Defaults to True.
        reject_nonfinite (bool, optional): Whether to skip outputs that have any non-finite values. Defaults to True.
    '''
    
    freq_upper_limit, cycles_at_freq_lim_lowest_Mc = binary_search_limiting_freq(t_, base_param_dict | {'Mc': Mc_limits[0], 'init_phi': 0}, phase_func, freq_limits)
    cycles_at_freq_lim_lowest_Mc /= (2 * np.pi)
    n_freq_grid = int(cycles_at_freq_lim_lowest_Mc * pts_per_cycle_freq)
    print(f'Generating frequency grid from {freq_limits[0]} to {freq_upper_limit} with {n_freq_grid} points (logspacing={freq_logspacing})')
    freq_limits = (freq_limits[0], freq_upper_limit)
    
    if freq_logspacing:
        init_freq = np.logspace(np.log10(freq_limits[0].to(u.Hz).value), np.log10(freq_limits[1].to(u.Hz).value), n_freq_grid) * u.Hz
    else:
        init_freq = np.linspace(freq_limits[0].to(u.Hz).value, freq_limits[1].to(u.Hz).value, n_freq_grid) * u.Hz
        
    param_list = []
    outputs = []
    if progress_bar:
        init_freq = tqdm.tqdm(init_freq)
    chirp_masses = []
    for i_f in init_freq:
        Mc_upper_limit, cycles_at_Mc_lim = binary_search_limiting_Mc(t_, base_param_dict | {'initial_freq': i_f, 'init_phi': 0}, phase_func, Mc_limits)
        cycles_at_Mc_lim -= phase_func(t_, base_param_dict | {'initial_freq': i_f, 'init_phi': 0, 'Mc': Mc_limits[0]})[-1]
        assert cycles_at_Mc_lim >= 0
        cycles_at_Mc_lim /= (2 * np.pi)
        n_Mc_grid = max(2, int(np.ceil(cycles_at_Mc_lim * pts_per_cycle_Mc)))
        if verbose:
            print(f'At {i_f}: Generating chirp mass grid from {Mc_limits[0]} to {Mc_upper_limit} with {n_Mc_grid} points (logspacing={Mc_logspacing})')
        Mc_limits = (Mc_limits[0], Mc_upper_limit)
        if Mc_logspacing:
            chirp_mass = np.logspace(np.log10(Mc_limits[0].to(u.Msun).value), np.log10(Mc_limits[1].to(u.Msun).value), n_Mc_grid) * u.Msun
        else:
            chirp_mass = np.linspace(Mc_limits[0].to(u.Msun).value, Mc_limits[1].to(u.Msun).value, n_Mc_grid) * u.Msun
            
        chirp_masses.extend([Mc.to(u.Msun).value for Mc in chirp_mass])
            
        for Mc in chirp_mass:
            params = base_param_dict.copy()
            params['initial_freq'] = i_f
            params['Mc'] = Mc
            try:
                out = gen_func(t_, params)
            except Exception:
                if skip_gen_errors:
                    outputs.append(np.full_like(out, np.nan))
                    param_list.append(params)
                    continue
                else:
                    print(f"Error generating output in grid occurred with params {params}:")
                    raise
            if reject_nonfinite and np.any(~np.isfinite(out)):
                outputs.append(np.full_like(out, np.nan))
            else:
                outputs.append(phase_func(t_, params))
            param_list.append(params)
    return (init_freq, np.unique(chirp_masses) * u.Msun), param_list, outputs

def binary_search_limiting_freq(t_, param_dict, phase_func, init_freq_limits):
    def test_func(freq):
        params = param_dict.copy()
        params['initial_freq'] = freq
        try:
            phase = phase_func(t_, params)
            return not np.isfinite(phase[-1])
        except Exception:
            return True
    
    freq_extent = init_freq_limits[1] - init_freq_limits[0]
    
    freq_threshold = isidora.utils.continuous_binary_search(freq_extent * 1e-6, init_freq_limits[0], init_freq_limits[1], test_func)
    threshold_params = param_dict.copy()
    threshold_params['initial_freq'] = freq_threshold
    return freq_threshold, phase_func(t_, threshold_params)[-1]

def binary_search_limiting_Mc(t_, param_dict, phase_func, init_Mc_limits):
    def test_func(Mc):
        params = param_dict.copy()
        params['Mc'] = Mc
        try:
            phase = phase_func(t_, params)
            return not np.isfinite(phase[-1])
        except Exception:
            return True

    Mc_extent = init_Mc_limits[1] - init_Mc_limits[0]

    Mc_threshold = isidora.utils.continuous_binary_search(Mc_extent * 1e-6, init_Mc_limits[0], init_Mc_limits[1], test_func)
    threshold_params = param_dict.copy()
    threshold_params['Mc'] = Mc_threshold

    return Mc_threshold, phase_func(t_, threshold_params)[-1]

def binary_search_limiting_Mc_freq_thresh(t_, param_dict, freq_func, thresh_freq, init_Mc_limits):
    def test_func(Mc):
        params = param_dict.copy()
        params['Mc'] = Mc
        try:
            freq = freq_func(t_, params, mask_meco=False)[-1]
            return freq <= thresh_freq
        except Exception:
            return True

    Mc_extent = init_Mc_limits[1] - init_Mc_limits[0]

    Mc_threshold = isidora.utils.continuous_binary_search(Mc_extent * 1e-6, init_Mc_limits[0], init_Mc_limits[1], test_func)
    threshold_params = param_dict.copy()
    threshold_params['Mc'] = Mc_threshold

    return Mc_threshold, freq_func(t_, threshold_params, mask_meco=False)[-1]