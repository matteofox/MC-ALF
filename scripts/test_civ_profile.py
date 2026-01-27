import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from astropy.convolution import convolve, Gaussian1DKernel

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcalf.routines import hires_fitter
import jax.numpy as jnp

# Constants
CCGS = 2.9979245e10

def standard_voigt(wave, N, z, b, wrest, f, gamma):
    cold = 10.0**N
    zp1 = z + 1.0
    w_cm = wave / 1e8
    wrest_cm = wrest / 1e8
    nujk = CCGS / wrest_cm
    dnu = (b * 1e5) / wrest_cm 
    avoigt = gamma / (4 * np.pi * dnu)
    uvoigt = ((CCGS / (w_cm / zp1)) - nujk) / dnu
    cne = 0.014971475 * cold * f
    tau = cne * wofz(uvoigt + 1j * avoigt).real / dnu
    return tau

def main():
    mock_dir = 'mockdata'
    if not os.path.exists(mock_dir):
        os.makedirs(mock_dir)

    z_target = 3.0
    N_target = 14.0
    b_target = 20.0
    
    # Spectrum parameters
    wave = np.linspace(6180, 6220, 2000)
    
    # Calculate velocity step (approx)
    # dv = c * dlam / lam
    c_kms = 299792.458
    dlam = wave[1] - wave[0]
    lam_ref = np.mean(wave)
    velstep = c_kms * dlam / lam_ref
    print(f"Calculated velstep: {velstep:.4f} km/s/pixel")
    
    # Fit config uses specres = 6.0 (assumed km/s FWHM)
    specres_kms = 6.0
    
    lines = [
        {'wrest': 1548.195, 'f': 0.190, 'gamma': 2.64e8},
        {'wrest': 1550.770, 'f': 0.0952, 'gamma': 2.63e8}
    ]
    
    tau_tot = np.zeros_like(wave)
    for l in lines:
        t = standard_voigt(wave, N_target, z_target, b_target, l['wrest'], l['f'], l['gamma'])
        tau_tot += t
        
    true_flux_unconvolved = np.exp(-tau_tot)
    
    # Convolve with instrument resolution
    sigma_pix = (specres_kms / 2.354820) / velstep
    print(f"Convolution kernel sigma: {sigma_pix:.4f} pixels")
    
    kernel = Gaussian1DKernel(stddev=sigma_pix)
    true_flux_convolved = convolve(true_flux_unconvolved, kernel, boundary='extend')
    
    # Add noise
    noise_level = 0.02
    noise = np.ones_like(wave) * noise_level
    observed_flux = true_flux_convolved + np.random.normal(0, noise_level, size=len(wave))
    
    specfile = os.path.join(mock_dir, 'civ_mock_spec.txt')
    with open(specfile, 'w') as f:
        f.write('Wave Flux Err\n')
        for w, fl, n in zip(wave, observed_flux, noise):
            f.write(f'{w} {fl} {n}\n')

    print(f"Created convolved mock spectrum {specfile}")
    
    # Plotting check
    plt.figure(figsize=(10, 6))
    plt.plot(wave, true_flux_unconvolved, label='Unconvolved Truth', alpha=0.5, linestyle=':')
    plt.plot(wave, observed_flux, label=f'Mock Data (Conv FWHM={specres_kms}km/s)', alpha=0.7)
    plt.xlabel('Wavelength')
    plt.legend()
    plt.savefig(os.path.join(mock_dir, 'mock_generation_check.png'))
    print("Saved check plot")

if __name__ == '__main__':
    main()
