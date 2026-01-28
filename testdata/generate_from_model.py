import numpy as np
import matplotlib.pyplot as plt
from mcalf.routines import hires_fitter
import sys

def main():
    # Model Parameters
    z_true = [3.0,3.001,3.0005,3.002]
    N_true = [13.8,13.5,13.2,13.4]
    b_true = [20.0,25.0,15.0,30.0]
    
    # Grid/Config from existing file
    specfile = 'civ_mock_spec.txt'
    fitlines = ['CIV 1548', 'CIV 1550']
    fitrange = [[6180, 6220]]
    specres = [8.0] # FWHM km/s

    flux_arr = []

    # Initialize Fitter to access model generation
    # We use debug=False to avoid unnecessary prints
    print(f"Initializing fitter with {specfile}...")
    with hires_fitter.als_fitter(specfile, fitrange=fitrange, 
                                 fitlines=fitlines, ncomp=[1,1], 
                                 debug=False, specres=specres) as fitter:
        
        for ii in range(len(N_true)):
         # Create Parameter Array
         # Structure: [continuum, N, z, b]
         # Continuum = 1.0 (Mock)
         p_true = np.array([1.0, N_true[ii], z_true[ii], b_true[ii]])

         print(f"Generating Model for N={N_true[ii]}, z={z_true[ii]}, b={b_true[ii]}, Res={specres[0]}")

         # Generate Flux (Convolved)
         # reconstruct_spec uses the internal stored specres or p[0] if freespecres
         # We passed specres=[8.0] to init, so it uses that.
         flux_arr.append(fitter.reconstruct_spec(p_true))


        flux_model = np.prod(np.array(flux_arr), axis=0)
        print(len(flux_model))

        # Get Wavelength Grid
        wave = fitter.obj_wl
        
        # Add Noise (SNR 50 => Sigma = 1/50 = 0.02)
        noise_level = 0.02
        np.random.seed(42)
        noise = np.random.normal(0, noise_level, size=len(wave))
        
        flux_mock = flux_model + noise
        err_mock = np.ones_like(flux_mock) * noise_level
        
        # Save to File
        # Format: Wave Flux Err
        header = "Wave Flux Err"
        data = np.column_stack([wave, flux_mock, err_mock])
        
        outfile = "civ_mock_spec_multicomp.txt"
        np.savetxt(outfile, data, header=header)
        print(f"Saved generated mock data to {outfile}")
        
        # Plot for verification
        plt.figure(figsize=(10, 6))
        plt.plot(wave, flux_mock, 'k-', lw=0.5, alpha=0.5, label='Mock Data (SNR=50)')
        plt.plot(wave, flux_model, 'r-', lw=1.5, label='True Model (MC-ALF)')
        plt.xlabel('Wavelength (A)')
        plt.ylabel('Normalized Flux')
        plt.title(f'MC-ALF Generated Mock: z={z_true}, logN={N_true}, b={b_true}, FWHM={specres[0]}', fontsize=8)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('civ_mock_spec_multicomp.png')
        print("Saved plot to civ_mock_spec.png")
        
        # Debug Stats
        print(f"Model Min Flux: {np.min(flux_model):.4f}")

if __name__ == '__main__':
    main()
