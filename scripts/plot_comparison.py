import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from mcalf.routines import hires_fitter

def main():
    specfile = 'mockdata/civ_mock_spec.txt'
    fitlines = ['CIV 1548', 'CIV 1550']
    fitrange = [[6180, 6220]]
    # FWHM = 8.0 km/s
    specres = [8.0]
    
    # True Parameters
    N_true = 13.8
    z_true = 3.0
    b_true = 15.0
    p_true = np.array([1.0, N_true, z_true, b_true])
    
    # Initialize Fitter
    # Note: specres passed here overrides internal defaults
    with hires_fitter.als_fitter(specfile, fitrange=fitrange, 
                                 fitlines=fitlines, ncomp=[1,1], 
                                 debug=False, specres=specres) as fitter:
        
        print("Comparing Models...")
        
        # 1. Scipy Model (using internal reconstruct_spec)
        # Note: reconstruct_spec accepts p in specific format
        # [continuum, N, z, b]
        flux_scipy = fitter.reconstruct_spec(p_true)
        
        # 2. JAX Model (using get_jax_model_func)
        jax_func = fitter.get_jax_model_func()
        # JAX func expects JAX array
        flux_jax = jax_func(jnp.array(p_true))
        
        # 3. Data
        wave = fitter.obj_wl
        flux_data = fitter.obj
        
        # Calculate residuals
        res_scipy = flux_scipy - flux_jax
        max_diff = np.max(np.abs(res_scipy))
        max_diff_idx = np.argmax(np.abs(res_scipy))
        max_diff_wl = wave[max_diff_idx]
        
        print(f"Stats:")
        print(f"Scipy: Min={np.min(flux_scipy):.4f}, Mean={np.mean(flux_scipy):.4f}")
        print(f"JAX:   Min={np.min(flux_jax):.4f},   Mean={np.mean(flux_jax):.4f}")
        print(f"Max Difference (Scipy - JAX): {max_diff:.3e} at {max_diff_wl:.2f} A")
        
        # Scipy vs Data Stats
        res_data = flux_scipy - flux_data
        data_diff_max = np.max(np.abs(res_data))
        data_diff_mean = np.mean(res_data)
        data_diff_std = np.std(res_data)
        
        print(f"\nScipy - Data Stats:")
        print(f"  Data Min: {np.min(flux_data):.4f}, Mean: {np.mean(flux_data):.4f}")
        print(f"  Max Diff: {data_diff_max:.4f}")
        print(f"  Mean Diff: {data_diff_mean:.4e}")
        print(f"  Std Diff: {data_diff_std:.4f} (Expected ~0.02)")

        print("\n--- Calibration Check ---")
        print(f"Fitter Velstep: {fitter.velstep:.4f} km/s")
        for i, l in enumerate(fitter.linepars):
            print(f"Line {i}: {l['name']}")
            print(f"  wrest: {l['wrest'].value:.4f}")
            print(f"  f:     {l['f']:.4f}")
            print(f"  gamma: {l['gamma'].value:.4e}")
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                 gridspec_kw={'height_ratios': [3, 1]})
        
        # Top Panel: Spectra
        ax = axes[0]
        ax.plot(wave, flux_data, 'k-', lw=0.5, alpha=0.3, label='Mock Data')
        ax.plot(wave, flux_scipy, 'b-', lw=1.5, alpha=0.7, label='Scipy Model (Truth)')
        ax.plot(wave, flux_jax, 'r--', lw=1.5, alpha=0.7, label='JAX Model (Truth)')
        
        ax.set_ylabel('Normalized Flux')
        ax.set_title(f"Model Comparison: z={z_true}, logN={N_true}, b={b_true}, Res={specres[0]}km/s")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom Panel: Residuals
        ax = axes[1]
        ax.plot(wave, res_scipy, 'g-', lw=1, label='Scipy - JAX')
        ax.set_ylabel('Difference')
        ax.set_xlabel('Wavelength (A)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1e-4, 1e-4) # Zoom in on errors
        
        outfile = 'mockdata/model_comparison.png'
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        print(f"Saved {outfile}")

if __name__ == '__main__':
    main()
