import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcalf.routines import hires_fitter
import jax.numpy as jnp

# Constants
CCGS = 2.9979245e10

def standard_voigt(wave, N, z, b, wrest, f, gamma):
    # wave in Angstroms
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
    # Setup directory
    mock_dir = 'mockdata'
    if not os.path.exists(mock_dir):
        os.makedirs(mock_dir)
        print(f"Created directory {mock_dir}")

    # 1. Define physics
    z_target = 3.0
    N_target = 14.0 # logN
    b_target = 20.0 # km/s
    
    # 2. Create True Spectrum using Standard Voigt
    wave = np.linspace(6180, 6220, 2000)
    
    # CIV params (approximate if linetools not loaded, but let's try to match linetools values if possible)
    # Using defaults from typical atomic data if linetools lookup fails or just hardcode for consistency
    # CIV 1548: 1548.195, f=0.190, gamma=2.64e8
    # CIV 1550: 1550.770, f=0.0952, gamma=2.63e8
    
    lines = [
        {'wrest': 1548.195, 'f': 0.190, 'gamma': 2.64e8},
        {'wrest': 1550.770, 'f': 0.0952, 'gamma': 2.63e8}
    ]
    
    tau_tot = np.zeros_like(wave)
    for l in lines:
        t = standard_voigt(wave, N_target, z_target, b_target, l['wrest'], l['f'], l['gamma'])
        tau_tot += t
        
    true_flux = np.exp(-tau_tot)
    
    # Add noise
    noise_level = 0.02
    noise = np.ones_like(wave) * noise_level
    observed_flux = true_flux + np.random.normal(0, noise_level, size=len(wave))
    
    specfile = os.path.join(mock_dir, 'civ_mock_spec.txt')
    with open(specfile, 'w') as f:
        f.write('Wave Flux Err\n')
        for w, fl, n in zip(wave, observed_flux, noise):
            f.write(f'{w} {fl} {n}\n')

    print(f"Created mock spectrum {specfile}")

    try:
        # 3. Initialize als_fitter with CIV doublet
        fitlines = ['CIV 1548', 'CIV 1550']
        fitrange = [[6180, 6220]]
        
        # Note: als_fitter will use linetools to look up atomic data. 
        # If linetools data differs slightly from my hardcoded values above, there might be a small shift.
        # But visually it should be close.
        
        with hires_fitter.als_fitter(specfile, fitrange=fitrange, 
                                     fitlines=fitlines, ncomp=[1,1], 
                                     debug=True, specres=[6.0]) as fitter:
            
            # 4. Get JAX model function
            print("Compiling JAX model function...")
            model_func = fitter.get_jax_model_func()
            
            # 5. Evaluate JAX Model
            # p vector: [ncomp, N, z, b, ... fillers ...]
            p = np.zeros(fitter.ndim)
            p[fitter.startind] = 1.0 # ncomp
            p[fitter.startind+1] = N_target
            p[fitter.startind+2] = z_target
            p[fitter.startind+3] = b_target
            
            print(f"Evaluating JAX model for z={z_target}, logN={N_target}, b={b_target}")
            
            jax_flux = model_func(jnp.array(p))
            
            # 6. Plot comparison
            plt.figure(figsize=(10, 6))
            # Use fitter.obj_wl for x-axis as als_fitter might trim or process data
            
            plt.plot(fitter.obj_wl, fitter.obj, label='Mock Data (Standard Voigt + Noise)', color='lightgray', lw=1)
            plt.plot(fitter.obj_wl, jax_flux, label='JAX Model', color='red', linestyle='--', lw=1.5)
            
            # Also plot the true flux (interpolated to fitter.obj_wl if needed)
            # Since standard_voigt was on 'wave', and fitter.obj_wl is subset of 'wave' (probably full set here)
            # We can just re-evaluate standard voigt on fitter.obj_wl for clean comparison line
            
            std_tau_check = np.zeros_like(fitter.obj_wl)
            # We need the ACTUAL parameters fitter loaded from linetools to be perfectly consistent for the check line
            # Checking fitter.linepars match lines above
            # But let's just plot the JAX one. If it sits on the data, it's good.
            
            plt.xlabel('Wavelength (A)')
            plt.ylabel('Normalized Flux')
            plt.title('Verification: Standard Voigt Data vs JAX Model')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(-0.1, 1.2)
            
            plot_file = os.path.join(mock_dir, 'civ_profile_comparison.png')
            plt.savefig(plot_file)
            print(f"Plot saved to {plot_file}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
