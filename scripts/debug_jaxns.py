import jax
import jax.numpy as jnp
from jaxns import NestedSampler, Model, Prior
import tensorflow_probability.substrates.jax as tfp
tfpd = tfp.distributions

def main():
    print("Testing JAXNS Model instantiation...")
    
    # Dummy parameters
    ndim = 3
    lowers = jnp.zeros(ndim)
    uppers = jnp.ones(ndim)
    
    # Option 1: Prior model using primitives
    def prior_model():
        # Define a single vector parameter 'theta'
        theta = yield Prior(tfpd.Uniform(low=lowers, high=uppers), name='theta')
        return theta

    def log_likelihood(theta):
        return -jnp.sum(theta**2)

    try:
        model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
        print("Model created successfully with generator prior structure.")
        
        ns = NestedSampler(model=model)
        print("NestedSampler initialized.")
        
        termination_reason, state = ns(key=jax.random.PRNGKey(0))
        print("Run successful.")
        
    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
