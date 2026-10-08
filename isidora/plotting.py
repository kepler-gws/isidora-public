import os
import sys
from dotenv import find_dotenv
sys.path.append(os.path.dirname(find_dotenv()))

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import ipywidgets as widgets
import matplotlib.colors as colors

import isidora.training
import isidora.waveform


def plot_max_freq_and_n_cycles(title, base_params, phase_func, freq_func, 
                               obs_time_max, obs_cadence,
                               freq_limits, Mc_limits,
                               plot_meco=False,
                               explicit_obs_time=None,
                               return_data=False,
                               **grid_kwargs):

    if explicit_obs_time is not None:
        obs_time = explicit_obs_time
    else:
        obs_time = np.arange(0, obs_time_max.to(u.s).value, obs_cadence.to(u.s).value) * u.s
    (freq, Mc), _, phase = isidora.training.freq_Mc_grid(obs_time, base_params, phase_func, freq_limits, Mc_limits, reject_nonfinite=True, freq_logspacing=True, Mc_logspacing=True, **grid_kwargs)
    n_cycles = [c[-1] / (2 * np.pi) for c in phase]
    del phase
    max_freq_in_obs = isidora.training.freq_Mc_grid(obs_time, base_params, freq_func, freq_limits, Mc_limits, reject_nonfinite=False, freq_logspacing=True, Mc_logspacing=True, **grid_kwargs)[-1]
    max_freq_in_obs = [f[-1].to(u.Hz).value for f in max_freq_in_obs]

    if plot_meco:
        meco_freqs = isidora.training.freq_Mc_grid(obs_time, base_params, lambda _, p: isidora.waveform.meco_gw_freq(p),
                                                   freq_limits, Mc_limits, reject_nonfinite=False, freq_logspacing=True, Mc_logspacing=True, **grid_kwargs)[-1]
        meco_freqs = [f.to(u.Hz).value for f in meco_freqs]

    grid_reshape = lambda l: np.array(l).reshape((len(freq), len(Mc))).T

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{title}')
    
    def postplot():
        plt.xlabel('Initial GW frequency (Hz)')
        plt.ylabel('Chirp mass (Msun)')
        plt.xscale('log')
        plt.yscale('log')
    def meco_plot():
        plt.contour(freq.to(u.Hz).value, Mc.to(u.Msun).value, grid_reshape(max_freq_in_obs) / grid_reshape(meco_freqs), levels=[0.95], 
                    colors='darkgreen', linestyles='-', label='0.95 x MECO frequency')
        # plt.legend()

    plt.sca(axes[0])
    plt.pcolormesh(freq.to(u.Hz).value, Mc.to(u.Msun).value, grid_reshape(n_cycles), norm=colors.LogNorm(), cmap='viridis')
    plt.colorbar(label=f'# cycles in observation period')
    if plot_meco:
        meco_plot()
    postplot()
    
    plt.sca(axes[1])
    plt.pcolormesh(freq.to(u.Hz).value, Mc.to(u.Msun).value, grid_reshape(max_freq_in_obs), norm=colors.LogNorm(), cmap='inferno')
    plt.colorbar(label=f'Maximum frequency in observation period (Hz)')
    if plot_meco:
        meco_plot()
    postplot()

    if return_data:
        return fig, axes, (freq, Mc, grid_reshape(n_cycles), grid_reshape(max_freq_in_obs))
    else:
        return fig, axes


def plot_phase_difference(title, base_params, baseline_phase_func, new_phase_func, 
                               obs_time_max, obs_cadence,
                               freq_limits, Mc_limits,
                               **grid_kwargs):
    
    obs_time = np.arange(0, obs_time_max.to(u.s).value, obs_cadence.to(u.s).value) * u.s
    (freq, Mc), _, baseline_phase = isidora.training.freq_Mc_grid(obs_time, base_params, baseline_phase_func, freq_limits, Mc_limits, reject_nonfinite=True, freq_logspacing=True, Mc_logspacing=True, **grid_kwargs)
    new_phase = isidora.training.freq_Mc_grid(obs_time, base_params, new_phase_func, freq_limits, Mc_limits, reject_nonfinite=True, freq_logspacing=True, Mc_logspacing=True, **grid_kwargs)[-1]
    
    baseline_phase = [c[-1] / (2 * np.pi) for c in baseline_phase]
    new_phase = [c[-1] / (2 * np.pi) for c in new_phase]
    
    grid_reshape = lambda l: np.array(l).reshape((len(freq), len(Mc))).T
    
    phase_diff = grid_reshape(new_phase) - grid_reshape(baseline_phase)

    plt.title(f'{title}')
    
    def postplot():
        plt.xlabel('Initial GW frequency (Hz)')
        plt.ylabel('Chirp mass (Msun)')
        plt.xscale('log')
        plt.yscale('log')

    plt.pcolormesh(freq.to(u.Hz).value, Mc.to(u.Msun).value, phase_diff, norm=colors.SymLogNorm(linthresh=1e-1), cmap='viridis')
    plt.colorbar(label=f'Cumulative cycle difference')
    postplot()