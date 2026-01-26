import sys
import os
import numpy as np
# Add parent dir to path to find mcalf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create dummy spec
with open('dummy_spec.txt', 'w') as f:
    f.write('Wave Flux Err\n')
    for w in np.linspace(1214, 1217, 100):
        f.write(f'{w} 1.0 0.1\n')

try:
    from mcalf.routines import hires_fitter
    import jax
    print("JAX is available.")
except ImportError as e:
    print(f"Dependency missing ({e}), skipping JAX tests.")
    sys.exit(0)

print("Initializing als_fitter...")

try:
    # Minimal init
    # We rely on linetools being present as per user environment
    with hires_fitter.als_fitter('dummy_spec.txt', fitrange=[[1214, 1217]], 
                                 fitlines=['HI 1215'], ncomp=[1,1], 
                                 debug=True) as fitter:
        
        print("Getting JAX likelihood...")
        log_prob = fitter.get_jax_likelihood()
        
        print("Compiling and evaluating...")
        # Mean parameter vector
        p = np.array([np.mean(b) for b in fitter.bounds])
        
        # JIT compilation
        val = log_prob(p)
        print(f"Log-likelihood value: {val}")
        
        assert np.isfinite(val)
        print("JAX Component Test Passed!")
        
        # Test gradient
        print("Testing Gradient...")
        grad_fun = jax.grad(log_prob)
        g = grad_fun(p)
        print(f"Gradient norm: {np.linalg.norm(g)}")
        assert np.all(np.isfinite(g))
        print("Gradient Test Passed!")

except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
finally:
    if os.path.exists('dummy_spec.txt'):
        os.remove('dummy_spec.txt')
