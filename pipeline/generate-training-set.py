import os
import sys
from dotenv import find_dotenv
sys.path.append(os.path.dirname(find_dotenv()))
import copy

import numpy as np
import astropy.units as u
import astropy.constants as const
import tqdm
import dill as pickle

import isidora.waveform
import isidora.training

OBS_TIME_MAX = 3.5 * u.yr
OBS_CADENCE = 30 * u.min

NYQUIST_FREQ = (2 * 30 * u.min)**-1

OBS_TIME = np.arange(0, OBS_TIME_MAX.to(u.s).value, OBS_CADENCE.to(u.s).value) * u.s
print(f'Observing at {len(OBS_TIME)} points')

BASE_GW_PARAMS = {'q': 1}

INIT_PHI = np.array([0]) # np.linspace(0, 2 * np.pi, 10)

# Notes: I would do restricted PN, but there's issues where the Newtonian amplitude goes singular before the frequency of the phenom model reaches MECO.
AMP_FUNC, PHASE_FUNC, FREQ_FUNC = isidora.waveform.amp_minus_dl_phenomIMRT, isidora.waveform.phase_phenomIMRT, isidora.waveform.freq_phenomIMRT

FULL_TILE_PREFIX = 'data/training_set_full'
VALIDATION_TILE_PREFIX = 'data/validation_set_full'

def filter_nan_nyquist(grid_gen_output):
    (freq, Mc), param_list, total_phase = grid_gen_output
    total_cycles = np.array([c[-1] / (2 * np.pi) for c in total_phase])

    print(f'{len(param_list)} waveforms in grid, before nan filtering')

    # Filter out points with nan cycles
    mask = ~np.isnan(total_cycles)
    param_list = [param for param, m in zip(param_list, mask) if m]
    total_cycles = total_cycles[mask]
    assert len(param_list) == len(total_cycles)

    print(f'{len(param_list)} waveforms in grid, after nan filtering')

    # Filter out points that have any times above the Nyquist frequency (of waveform)
    training_set_peak_freqs = np.array([FREQ_FUNC(OBS_TIME, p | {'init_phi': 0})[-1].to(u.Hz).value for p in param_list])
    mask = training_set_peak_freqs < NYQUIST_FREQ.to(u.Hz).value
    param_list = [param for param, m in zip(param_list, mask) if m]
    total_cycles = total_cycles[mask]
    training_set_peak_freqs = training_set_peak_freqs[mask]
    assert len(param_list) == len(total_cycles)

    print(f'{len(param_list)} waveforms in grid, after Nyquist cutoff filtering')
    
    return param_list, total_cycles

def write_waveforms_to_memmap(param_list, tile, file_path, skip_het=True):
    wf_file = np.lib.format.open_memmap(file_path, dtype=np.complex128, mode='w+', shape=(len(param_list), len(OBS_TIME)))
    for i, param in tqdm.tqdm(enumerate(param_list), total=len(param_list)):
        wf = tile.generate_full_waveform(param | {'init_phi': 0}, skip_het=skip_het)
        wf_file[i] = wf
    wf_file.flush()

if __name__ == "__main__":
    
    print('--Generating training set--')
    
    DYNAMIC_GRID_PARAMS = {
        'freq_limits': (1e-8 * u.Hz, 1e-5 * u.Hz),
        'Mc_limits':(1e7 * u.Msun, 1e10 * u.Msun),
        'freq_logspacing': False,
        'Mc_logspacing': False,
        'pts_per_cycle_freq': 0.5,
        'pts_per_cycle_Mc': 0.5,
    }

    #(freq, Mc), param_list, total_phase = isidora.training.freq_Mc_grid_dynamic(OBS_TIME[-2:], BASE_GW_PARAMS | {'init_phi': 0}, PHASE_FUNC, PHASE_FUNC, skip_gen_errors=True, verbose=True, **DYNAMIC_GRID_PARAMS)
    training_param_list, training_total_cycles = filter_nan_nyquist(isidora.training.freq_Mc_grid_dynamic(OBS_TIME[-2:], BASE_GW_PARAMS | {'init_phi': 0}, PHASE_FUNC, PHASE_FUNC, skip_gen_errors=True, verbose=True, **DYNAMIC_GRID_PARAMS))

    training_full_tile = isidora.training.HeterodyneFreqTile(tile_freq_limits=DYNAMIC_GRID_PARAMS['freq_limits'], 
                                                    tile_ref_gw_params=None, tile_ref_gw_cycles=0, tile_full_gw_param_list=training_param_list,
                                                    amp_func=AMP_FUNC, phase_func=PHASE_FUNC, freq_func=FREQ_FUNC,
                                                    physical_time=OBS_TIME)
    
    with open(f'{FULL_TILE_PREFIX}_tile.pkl', 'wb') as f:
        pickle.dump(training_full_tile, f)
        
    # write_waveforms_to_memmap(training_param_list, training_full_tile, f'{FULL_TILE_PREFIX}_wf.npy', skip_het=True)
    
    print('--Generating validation set--')
    
    # Generate validation waveform set.
    validation_grid_params = copy.deepcopy(DYNAMIC_GRID_PARAMS)
    validation_grid_params['pts_per_cycle_freq'] *= 2
    validation_grid_params['pts_per_cycle_Mc'] *= 2
    
    validation_param_list, validation_total_cycles = filter_nan_nyquist(isidora.training.freq_Mc_grid_dynamic(OBS_TIME[-2:], BASE_GW_PARAMS | {'init_phi': 0}, PHASE_FUNC, PHASE_FUNC, skip_gen_errors=True, verbose=True, **validation_grid_params))
    
    freq_grouped_training_param = isidora.training.group_param_list(training_param_list, 'initial_freq')[0]
    freq_grouped_validation_param = isidora.training.group_param_list(validation_param_list, 'initial_freq')[0]
    
    training_freq_limits = (min(freq_grouped_training_param.keys()), max(freq_grouped_training_param.keys()))
    # Cut columns outside training param set.
    freq_grouped_validation_param = {k: v for k, v in freq_grouped_validation_param.items() if training_freq_limits[0] <= k <= training_freq_limits[1]}

    n_filtered_in_training = 0
    
    filtered_validation_param_list = []
    for freq, freq_column_list in freq_grouped_validation_param.items():        
        # Two cases: either this frequency is in the training set, or it isn't.
        # In the first case, cut any validation points that are outside the training Mc range for the corresponding column.
        # In the second case, look at the two nearest columns in the training set in frequency. Take the max of the min Mc for the lower bound, and the min of the max Mc for the upper bound. Cut any validation points outside that range.
        if freq in freq_grouped_training_param:
            training_Mc_column = freq_grouped_training_param[freq]
            training_Mc_limits = (min([p['Mc'] for p in training_Mc_column]), max([p['Mc'] for p in training_Mc_column]))
            filtered_col = [p for p in freq_column_list if training_Mc_limits[0] <= p['Mc'] <= training_Mc_limits[1]]
            max_Mc_in_column = max([p['Mc'] for p in filtered_col])
            # Just for good measure, remove the maximum Mc in each column to make sure we're within the convex hull of the training set.
            filtered_col = [p for p in filtered_col if p['Mc'] < max_Mc_in_column]
            # Finally, filter out waveforms that were originally in the training set.
            for validation_p in filtered_col:
                for training_p in training_Mc_column:
                    if np.isclose(validation_p['initial_freq'], training_p['initial_freq']) and np.isclose(validation_p['Mc'], training_p['Mc']):
                        n_filtered_in_training += 1
                        continue
                filtered_validation_param_list.append(validation_p)
        else:
            # Find nearest training frequencies.
            training_freqs = u.Quantity(list(freq_grouped_training_param.keys()), unit=u.Hz)
            lower_freqs = training_freqs[training_freqs < freq]
            upper_freqs = training_freqs[training_freqs > freq]
            if len(lower_freqs) == 0 or len(upper_freqs) == 0:
                continue
            lower_nearest_freq = max(lower_freqs)
            upper_nearest_freq = min(upper_freqs)
            lower_nearest_Mc_column = freq_grouped_training_param[lower_nearest_freq]
            upper_nearest_Mc_column = freq_grouped_training_param[upper_nearest_freq]
            lower_nearest_Mc_limits = (min([p['Mc'] for p in lower_nearest_Mc_column]), max([p['Mc'] for p in lower_nearest_Mc_column]))
            upper_nearest_Mc_limits = (min([p['Mc'] for p in upper_nearest_Mc_column]), max([p['Mc'] for p in upper_nearest_Mc_column]))
            Mc_lower_limit = max(lower_nearest_Mc_limits[0], upper_nearest_Mc_limits[0])
            Mc_upper_limit = min(lower_nearest_Mc_limits[1], upper_nearest_Mc_limits[1])
            filtered_col = [p for p in freq_column_list if Mc_lower_limit <= p['Mc'] <= Mc_upper_limit]
            max_Mc_in_column = max([p['Mc'] for p in filtered_col])
            # Just for good measure, remove the maximum Mc in each column to make sure we're within the convex hull of the training set.
            filtered_validation_param_list += [p for p in filtered_col if p['Mc'] < max_Mc_in_column]

    print(f'Filtered out {n_filtered_in_training} points that were in training set')
    print('After filtering validation points outside training set convex hull, there are {} validation points'.format(len(filtered_validation_param_list)))
            
    validation_full_tile = isidora.training.HeterodyneFreqTile(tile_freq_limits=validation_grid_params['freq_limits'], 
                                                    tile_ref_gw_params=None, tile_ref_gw_cycles=0, tile_full_gw_param_list=filtered_validation_param_list,
                                                    amp_func=AMP_FUNC, phase_func=PHASE_FUNC, freq_func=FREQ_FUNC,
                                                    physical_time=OBS_TIME)
    with open(f'{VALIDATION_TILE_PREFIX}_tile.pkl', 'wb') as f:
        pickle.dump(validation_full_tile, f)
            
    write_waveforms_to_memmap(filtered_validation_param_list, validation_full_tile, f'{VALIDATION_TILE_PREFIX}_wf.npy', skip_het=True)