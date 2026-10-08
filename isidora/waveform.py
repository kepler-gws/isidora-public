import itertools
import functools
import os

import numpy as np
import astropy.units as u
import astropy.constants as const
import scipy.optimize
os.environ["PHENOMXPY_CONSTANTS"] = "astropy"
import phenomxpy
from pycbc.pnutils import meco_frequency
# import lalsimulation

"""
Design notes:

Want to keep this flexible, just assume the stationary-phase approximation: signal is separable into A(t) * exp(i * phi(t)),
where A(t) and phi(t) also depend on some generic `params` dictionary.

Param names (dictionary keys):
  - initial_freq: non-angular frequency of the GW at t=0. (Hz)
  - Mc: chirp mass of the binary (Msun)
  - q: mass ratio of binary (dimensionless)
  - init_phi: initial phase of the GW (dimensionless)
-------
Derived quantities:
  - sym_mass_ratio: symmetric mass ratio in [0, 1/4] (dimensionless). Refered to with greek letters nu, or eta. Given typo potential, given verbose name here.
  - mu: reduced mass of the binary (Msun)
  - total_M: total mass of the binary (Msun)
  - rel_delta_M: relative mass difference (dimensionless)
  - x_freq: dimensionless frequency parameter used in PN expansions (dimensionless)
"""

"""General"""
# Schwarzschild ISCO frequency.
def isco_gw_freq(total_M):
    return const.c**3 / (6**(3/2) * np.pi * const.G * total_M)

# Border between early and late inspiral.
def meco_gw_freq(params):
    sec_params = secondary_params(params)
    return meco_frequency(sec_params['m_1'].to(u.Msun).value, sec_params['m_2'].to(u.Msun).value, 0, 0) * u.Hz

"""Newtonian approximation"""

def mass_strain_freq_to_dist_0pn(freq, Mc, strain):
    return ((2 * const.G**(5/3) / const.c**4) * (np.pi * freq)**(2/3) * Mc**(5/3) / strain).to(u.Mpc)

def mass_dist_freq_to_strain_0pn(freq, Mc, dl):
    return ((2 * const.G**(5/3) / const.c**4) * (np.pi * freq)**(2/3) * Mc**(5/3) / dl).to(u.dimensionless_unscaled).value

def gw_freq_0pn(chirp_mass, initial_freq, t_):
    '''Generate 0PN GW (not orbital!) frequency from chirp mass, initial frequency, and time.

    Args:
        chirp_mass (quantity.mass): Chirp mass of binary.
        initial_freq (quantity.frequency): Initial frequency of GW.
        t_ (quantity.time): Time at which to evaluate the phase.

    Returns:
        GW frequency (radian).
    '''
    coeff = (96/5) * np.pi**(8/3) * (chirp_mass * const.G / const.c**3)**(5/3)
    return ((-8/3) * coeff * t_ + initial_freq**(-8/3))**(-3/8)

# From pen-and-paper integration of the frequency function.
def gw_phase_0pn(chirp_mass, initial_freq, t_):
    coeff = (96/5) * np.pi**(8/3) * (chirp_mass * const.G / const.c**3)**(5/3)
    phase_func = lambda t: -(6 * np.pi / (5 * coeff)) * (((-8/3) * coeff * t + initial_freq**(-8/3))**(5/8) - initial_freq**(-5/3))
    return phase_func(t_) - phase_func(0 * u.s)

def max_Mc_before_merger_0pn(initial_freq, t_, starting_Mc=1e7, max_cutoff_Mc=1e10):
    t_max = np.max(t_)
    test_Mc_grid = np.logspace(np.log10(starting_Mc), np.log10(max_cutoff_Mc), 1000)
    for test_Mc in test_Mc_grid:
        final_freq = gw_freq_0pn(test_Mc * u.Msun, initial_freq * u.Hz, t_max)
        if not np.isfinite(final_freq):
            return test_Mc
    return max_cutoff_Mc

# Reference for these is Maggiore GW textbook section 4.1. No cos(inc) multiplicative factor since that's linearized out.
# This covers both h-plus and h-cross, since the only difference between the two at Newtonian order is an initial phase shift of pi/2.
def phase_newtonian(t_, params):
    freq, init_phi, Mc = params['initial_freq'], params['init_phi'], params['Mc']
    return gw_phase_0pn(Mc, freq, t_) + init_phi

def freq_newtonian(t_, params):
    freq, Mc = params['initial_freq'], params['Mc']
    return gw_freq_0pn(Mc, freq, t_)

def amp_minus_dl_newtonian(t_, params):
    freq, Mc = params['initial_freq'], params['Mc']
    dl = mass_strain_freq_to_dist_0pn(freq, Mc, 1)
    return mass_dist_freq_to_strain_0pn(gw_freq_0pn(Mc, freq, t_), Mc, dl)

def secondary_params(params):
    sec_params = {}
    sec_params['sym_mass_ratio'] = params['q'] / (1 + params['q'])**2
    sec_params['total_M'] = params['Mc'] * sec_params['sym_mass_ratio']**(-3/5)
    sec_params['mu'] = sec_params['sym_mass_ratio'] * sec_params['total_M']
    sec_params['rel_delta_M'] = (params['q'] - 1) / (params['q'] + 1)
    sec_params['m_1'] = sec_params['total_M'] * (params['q'] / (1 + params['q']))
    sec_params['m_2'] = sec_params['total_M'] - sec_params['m_1']
    return sec_params

"""PhenomIMRT: https://arxiv.org/abs/2004.08302
Notes: we just use the phase and amplitude up until the MECO frequency; the formalism used for this is an extension of TaylorT3 with additional 'pseudo-PN' terms calibrated to NR,
since TaylorT3 fails before MECO.
Despite what the paper says, the cut between inspiral and intermediate (late inspiral + merger?) is hardcoded to -150M, not t_MECO.
So we need to use the imr_ functions, and do the cut manually at t_MECO ourselves afterwards.
"""
def _setup_phenomIMRT_obj(params):
    sec_params = secondary_params(params)
    # We convert into NR units internally since physical units (supplying total mass to the model)
    # are buggy.
    return phenomxpy.IMRPhenomT(
        eta=sec_params['sym_mass_ratio'],
        s1=np.zeros(3), s2=np.zeros(3), # No spin
        f_min=phenomxpy.utils.HztoMf(params['initial_freq'].to(u.Hz).value, sec_params['total_M'].to(u.Msun).value),
        f_ref=phenomxpy.utils.HztoMf(params['initial_freq'].to(u.Hz).value, sec_params['total_M'].to(u.Msun).value),
        # delta_t=(30 * u.minute).to(u.s).value,
        # phi_ref=params['init_phi'], # Note: this is not the phase at f_ref, but a spatial angle (something related to Ylm harmonics)
        # eccentricity=0,
        # inclination=0 # face-on
        # atol=1e-324
        atol=1e-99
    )
    
def _t_offset_phenomIMRT(phenom_obj, params):
    # tref is a negative number (0 is time of coalescence)
    return phenom_obj.pWF.tref

def _setup_common_phenomIMRT(t_, params):
    sec_params = secondary_params(params)
    phenom = _setup_phenomIMRT_obj(params)
    t_rel_nr = phenomxpy.utils.SecondtoMass(t_.to(u.s).value, sec_params['total_M'].to(u.Msun).value) + _t_offset_phenomIMRT(phenom, params)
    return sec_params, phenom, t_rel_nr
    
def _before_meco_mask_phenomIMRT(phenom_obj, t_nr, params):
    sec_params = secondary_params(params)
    t_meco_nr = phenom_obj.pPhase._get_time_of_freq(phenomxpy.utils.HztoMf(meco_gw_freq(params).to(u.Hz).value, sec_params['total_M'].to(u.Msun).value))
    return t_nr <= t_meco_nr

def freq_phenomIMRT(t_, params, mask_meco=True):
    sec_params, phenom, t_rel_nr = _setup_common_phenomIMRT(t_, params)
    omega = phenom.pPhase.imr_omega(t_rel_nr)
    if mask_meco:
        omega[~_before_meco_mask_phenomIMRT(phenom, t_rel_nr, params)] = np.nan
    return (phenomxpy.utils.MftoHz(omega, sec_params['total_M'].to(u.Msun).value) / (2 * np.pi)) * u.Hz

def phase_phenomIMRT(t_, params, mask_meco=True):
    sec_params, phenom, t_rel_nr = _setup_common_phenomIMRT(t_, params)
    phase = phenom.pPhase.imr_phase(t_rel_nr)
    if mask_meco:
        phase[~_before_meco_mask_phenomIMRT(phenom, t_rel_nr, params)] = np.nan
    return phase - phenom.pPhase.inspiral_ansatz_phase(phenom.pWF.tref) + params['init_phi']

def amp_minus_dl_phenomIMRT(t_, params, mask_meco=True):
    sec_params, phenom, t_rel_nr = _setup_common_phenomIMRT(t_, params)
    # TODO: is this distance or luminosity distance? Docs unclear; assuming lum. dist for now.
    dist = mass_strain_freq_to_dist_0pn(params['initial_freq'], params['Mc'], 1)
    amp = phenomxpy.utils.AmpNRtoSI(phenom.pAmp.imr_amplitude(t_rel_nr), dist.to(u.Mpc).value, sec_params['total_M'].to(u.Msun).value)
    # TODO: why is this amplitude off by a factor of ~pi from Newtonian? Unclear but just fudging it for now.
    amp /= np.pi
    if mask_meco:
        amp[~_before_meco_mask_phenomIMRT(phenom, t_rel_nr, params)] = np.nan
    return amp

"""3.5PN approximation. Reference is Blanchet et al. 2014"""
"""4/20/2026: comment this out b/c non-trivial correctness checking + with this parameterization,
   the time of coalescence has to be found every call using root-finding algs, which is slow for PE.
   The correct approach is to re-implement LALSimulation's TaylorT3 in JAX, but this should really be done
   in `ripple`, and needs time-domain support for that.
   Edit: well, LALSimulation also has to do root-finding for the reference time/phase, but the argument that we should
   do it in ripple is still valid.
   For now, let's just use the orbit evolution algorithm in LALSimulation TaylorT3, which returns both the raw phase and also
   the parameter v = (pi * M * f)^(1/3), where M is the total mass.
   Reference for TaylorT3 is https://doi.org/10.1103/PhysRevD.80.084043 .
   Edit: uncommenting since TaylorT3 doesn't work either (see below). TODO: reimplement TaylorT3 in JAX with more sane units if needed.
   
   Update 4/22/2026: See https://arxiv.org/abs/2004.08302, TaylorT3 fails well before MECO too. Use IMRPhenomT instead, from phenomxpy package.
# For numerical overflow/underflow avoidance.
g_over_c3 = (const.G / const.c**3).to(u.s / u.Msun)
   
# Convert between non-angular GW frequency and dimensionless frequency parameter used in PN expansions.
def freq_to_x(gw_freq, total_M):
    return ((total_M * np.pi * gw_freq * g_over_c3)**(2/3)).to(u.dimensionless_unscaled)
def x_to_freq(x, total_M):
    return (x**(3/2) / (g_over_c3 * total_M * np.pi)).to(u.Hz)

# Just for one time value; formally, Theta is (t_c - t) * (scale factor)
def t_to_Theta_scale_fac(sym_mass_ratio, total_M):
    return (sym_mass_ratio / (5 * g_over_c3 * total_M)).to(u.s**(-1))

# The initial_freq param is defined to be the GW frequency at t=0. Since x_35pn gives x(Theta), at t=0 the root of x_35pn(Theta) = freq_to_x(initial_freq) gives us the coalescence Theta.
# Memoize just in case the root finder is non-deterministic?
@functools.cache
def Theta_coal(initial_freq, total_M, sym_mass_ratio):
    initial_x = freq_to_x(initial_freq, total_M)
    newtonian_Theta_coal = (4 * initial_x)**(-4)
    root_result = scipy.optimize.root_scalar(lambda Theta: (_x_35pn(sym_mass_ratio, Theta) - initial_x).to(u.dimensionless_unscaled), 
                                             x0=newtonian_Theta_coal, method='newton')
    if not root_result.converged:
        raise RuntimeError('Root finding for coalescence time did not converge.')
    return root_result.root

# Blanchet et al. 2014 equation 316
def _x_35pn(sym_mass_ratio, Theta):
    nu = sym_mass_ratio
    global_scale_fac = (1/4)
    # Do the adding in the exponent for numerical stability.
    global_theta_pow = -1/4
    pn_sum = (Theta**global_theta_pow + (743/4032 + 11*nu/48) * Theta**(global_theta_pow + -1/4) - (np.pi/5) * Theta**(global_theta_pow + -3/8)
              + (19583/254016 + 24401*nu/193536 + 31*(nu**2)/288) * Theta**(global_theta_pow + -1/2)
              + (-11891/53760 + 109*nu/1920) * np.pi * Theta**(global_theta_pow + -5/8)
              + (-10052469856691/6008596070400 + (np.pi**2)/6 + 107*np.euler_gamma/420 - 107*np.log(Theta/256)/3360 + (3147553127/780337152 - 451*(np.pi**2)/3072)*nu - (15211/442368)*nu**2 + (25565/331776)*nu**3) * Theta**(global_theta_pow + -3/4)
              + (-113868647/433520640 - (31821/143360)*nu + (294941/3870720)*nu**2) * np.pi * Theta**(global_theta_pow + -7/8))
    return global_scale_fac * pn_sum

def freq_35pn(t_, params):
    sec_params = secondary_params(params)
    Theta_coal_ = Theta_coal(params['initial_freq'], sec_params['total_M'], sec_params['sym_mass_ratio'])
    print(f'Coalescence Theta: {(Theta_coal_ / t_to_Theta_scale_fac(sec_params["sym_mass_ratio"], sec_params["total_M"])).to(u.yr)}')
    Theta_scale_fac = t_to_Theta_scale_fac(sec_params['sym_mass_ratio'], sec_params['total_M'])
    Theta = (Theta_coal_ - Theta_scale_fac * t_).to(u.dimensionless_unscaled)
    print(Theta.max())
    x_out = _x_35pn(sec_params['sym_mass_ratio'], Theta)
    # x is of order O(v^2/c^2), so for PN to be valid it should be << 1.
    print(f'Max x: {np.nanmax(x_out)}')
    return x_to_freq(x_out, sec_params['total_M'])

# TODO: double-check this transcription
def _phase_35pn_free(sym_mass_ratio, Theta, Theta_integration_const):
    nu = sym_mass_ratio    
    global_scale_fac = -(1/nu)
    # Do the adding in the exponent for numerical stability.
    global_theta_pow = 5/8
    pn_sum = (Theta**global_theta_pow + (3715/8064 + (55/96)*nu) * Theta**(global_theta_pow + -1/4) - (3*np.pi/4)*Theta**(global_theta_pow + -3/8)
              + (9275495/14450688 + (284875/258048)*nu + (1855/2048)*nu**2) * Theta**(global_theta_pow + -1/2)
              + (-38645/172032 + (65/2048)*nu) * np.pi * Theta**(global_theta_pow + -5/8) * np.log(Theta/Theta_integration_const)
              + (831032450749357/57682522275840 - (53/40)*np.pi**2 - (107/56)*np.euler_gamma + (107/448)*np.log(Theta/256) + (-126510089885/4161798144 + (2255/2048)*np.pi**2)*nu + (154565/1835008)*nu**2 - (1179625/1769472)*nu**3) * Theta**(global_theta_pow + -3/4)
              + ((188516689/173408256)/_x_35pn(nu, Theta) + (488825/516096)*nu - (141769/516096)*nu**2) * np.pi * Theta**(global_theta_pow + -7/8))
    return global_scale_fac * pn_sum

def phase_35pn(t_, params):
    '''
    Notes: the 3.5PN expansion given in Blanchet et al. 2014 equation 317 is in terms of an integration constant Theta_integration_const, which has to be found in terms of the initial observed frequency and initial phase.
    Under the conventions we use for this work, we define the initial phase at t = 0 (start of data, where frequency is initial_freq) to be init_phi.
    This means we have to solve the equation init_phi = _phase_35pn_free(nu, Theta_coal, Theta_integration_const) for Theta_integration_const, which we can do with a root finder.
    '''
    sec_params = secondary_params(params)
    
    Theta = t_ * t_to_Theta_scale_fac(sec_params['sym_mass_ratio'], sec_params['total_M'])
    nu = sec_params['sym_mass_ratio']
"""

"""4/20/2026: ugh I think this overflows in m_1 and m_2 because they're in kilograms for whatever reason. So we can't use the LALSuite bindings or PyCBC (which both depend on them) directly...
def _freq_phase_lal_taylorT3(t_, params):
    sec_params = secondary_params(params)
    # LALSimulation only accepts even time steps, so need to validate that 1. t_ is an array and 2. it has even spacing.
    assert isinstance(t_, np.ndarray), 't_ must be a numpy array.'
    delta_t = t_[1] - t_[0]
    assert np.allclose(np.diff(t_), delta_t), 't_ must have even spacing.'
    m_1 = sec_params['total_M'] * (params['q'] / (1 + params['q']))
    m_2 = sec_params['total_M'] - m_1
    V, phi = lalsimulation.SimInspiralTaylorT3PNEvolveOrbit(
        params['init_phi'], # phiRef at fRef (==f_min==initial_freq for us)
        delta_t.to(u.s).value, # deltaT
        m_1.to(u.kg).value, # m1
        m_2.to(u.kg).value, # m2
        params['initial_freq'].to(u.Hz).value, # f_min
        params['initial_freq'].to(u.Hz).value, # fRef (there's special-casing in the code for fRef == f_min)
        0, 0, # tidal deformability params, turn off here
        0, # tidal order flags. 0 = 0PN = off? But shouldn't matter since 0 tidal deformability
        7 # twice of max PN order, so this is 3.5PN
    )
    V, phi = V.data.data, phi.data.data
    freq = (V**3 / (np.pi * sec_params['total_M'].to(u.Msun) * const.G / const.c**3)).to(u.Hz)
    assert len(freq) == len(phi)
    lal_out_len = len(freq)
    # Length of returned arrays can be longer or shorter than supplied t_ array. Fill with nans or cut as necessary;
    # in LALSimulation source code, t = 0 is where f_min is (so lines up with our assumptions).
    output_freq, output_phi = np.full_like(t_, np.nan), np.full_like(t_, np.nan)
    if lal_out_len > len(t_):
        output_freq, output_phi = freq[:len(t_)], phi[:len(t_)]
    else:
        output_freq[:lal_out_len], output_phi[:lal_out_len] = freq.value, phi
    return output_freq, output_phi

def freq_lal_taylorT3(t_, params):
    return _freq_phase_lal_taylorT3(t_, params)[0]

def phase_lal_taylorT3(t_, params):
    return _freq_phase_lal_taylorT3(t_, params)[1]
"""