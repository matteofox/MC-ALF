import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from mcalf.routines import hires_fitter

def main():
    specfile = 'mockdata/civ_mock_spec.txt'
    
    # Setup parameters
    fitlines = ['CIV 1548', 'CIV 1550']
    fitrange = [[6180, 6220]]
    specres_low = [0.1] # Disable convolution
    
    # Parameters
    N = 14.0
    z = 3.0
    b = 20.0
    
    with hires_fitter.als_fitter(specfile, fitrange=fitrange, 
                                 fitlines=fitlines, ncomp=[1,1], 
                                 debug=False, specres=specres_low) as fitter:
        
        # 1. Original Scipy Model
        # We need to construct 'p' array for original reconstruct_spec
        # Original params: [ncomp, N, b, z] or [ncomp, N, z, b]?
        # Earlier investigation: [ncomp, N, z, b]
        
        # NOTE: reconstruct_spec takes 'p'.
        # p structure:
        # [ncomp (val), N, z, b]
        # startind usually 0.
        
        p_orig = np.zeros(10) # Safe size
        p_orig[0] = 1.0 # ncomp
        p_orig[1] = N
        p_orig[2] = z
        p_orig[3] = b 
        
        # Check order again to be absolutely sure
        # In reconstruct_spec:
        # _N, _z, _b = p[1+3*i:1+3*i+3]
        # So N, z, b.
        
        flux_orig = fitter.reconstruct_spec(p_orig)
        
        # 2. JAX Model
        # get_jax_model_func returns a function that takes 'p'
        # My JAX implementation assumes SAME p structure [N, z, b]
        
        jax_func = fitter.get_jax_model_func()
        flux_jax = jax_func(jnp.array(p_orig))
        
        # Comparison
        diff = flux_orig - flux_jax
        
        idx_orig = np.argmin(flux_orig)
        idx_jax = np.argmin(flux_jax)
        
        wl_orig = fitter.obj_wl[idx_orig]
        wl_jax = fitter.obj_wl[idx_jax]
        
        print(f"Original Flux Min: {flux_orig.min()} at {wl_orig:.4f} A")
        print(f"JAX Flux Min:      {flux_jax.min()} at {wl_jax:.4f} A")
        print(f"Shift:             {wl_orig - wl_jax:.4f} A")
        idx_diff = np.argmax(np.abs(diff))
        wl_diff = fitter.obj_wl[idx_diff]
        
        print(f"Max Difference:    {np.max(np.abs(diff))} at {wl_diff:.4f} A")
        print(f"Orig Flux at MaxDiff: {flux_orig[idx_diff]}")
        print(f"JAX Flux at MaxDiff:  {flux_jax[idx_diff]}")
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(fitter.obj_wl, flux_orig, 'k-', label='Original (Scipy)', lw=2)
        plt.plot(fitter.obj_wl, flux_jax, 'r--', label='JAX', lw=2)
        plt.legend()
        plt.title(f'Voigt Comparison (N={N}, b={b}, z={z})')
        plt.ylabel('Normalized Flux')
        
        plt.subplot(2, 1, 2)
        plt.plot(fitter.obj_wl, diff, 'b-')
        plt.ylabel('Residual (Orig - JAX)')
        plt.xlabel('Wavelength (A)')
        
        plt.tight_layout()
        plt.savefig('voigt_comparison.png')
        print("Saved voigt_comparison.png")

if __name__ == '__main__':
    main()
