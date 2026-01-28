import jax.numpy as jnp
from jax import jit

@jit
def erfcx(x):
    """Scaled complementary error function exp(-x*x) based on Shepherd and Laframboise (1981)
    
    Args:
    x: should be larger than -9.3

    Returns:
        jnp.array: Scaled complementary error function exp(-x*x)
    """
    a = jnp.abs(x)
    b = (a - 2.0) / (a + 2.0)
    q = (-a * b - 2.0 * (b + 1.0) + a) / (a + 2.0) + b
    p = (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (5.92470169e-5 * q + 1.61224554e-4) * q
                                        - 3.46481771e-4
                                    )
                                    * q
                                    - 1.39681227e-3
                                )
                                * q
                                + 1.20588380e-3
                            )
                            * q
                            + 8.69014394e-3
                        )
                        * q
                        - 8.01387429e-3
                    )
                    * q
                    - 5.42122945e-2
                )
                * q
                + 1.64048523e-1
            )
            * q
            - 1.66031078e-1
        )
        * q
        - 9.27637145e-2
    ) * q + 2.76978403e-1

    q = (p + 1.0) / (1.0 + 2.0 * a)
    d = (p + 1.0) - q * (1.0 + 2.0 * a)
    f = 0.5 * d / (a + 0.5) + q
    f = jnp.where(x >= 0.0, f, 2.0 * jnp.exp(x**2) - f)

    return jnp.float32(f)

# Constants for Algorithm 916
an = jnp.array(
    [
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
        8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5,
    ], dtype=jnp.float32
)

a2n2 = jnp.array(
    [
        0.25, 1.0, 2.25, 4.0, 6.25, 9.0, 12.25, 16.0, 20.25, 25.0, 30.25, 36.0,
        42.25, 49.0, 56.25, 64.0, 72.25, 81.0, 90.25, 100.0, 110.25, 121.0, 132.25,
        144.0, 156.25, 169.0, 182.25,
    ], dtype=jnp.float32
)

def faddeeva_sigma1(exx, y2):
    Sigma1 = exx * (
        7.78800786e-01 / (0.25 + y2)
        + 3.67879450e-01 / (1.0 + y2)
        + 1.05399221e-01 / (2.25 + y2)
        + 1.83156393e-02 / (4.0 + y2)
        + 1.93045416e-03 / (6.25 + y2)
        + 1.23409802e-04 / (9.0 + y2)
        + 4.78511765e-06 / (12.25 + y2)
        + 1.12535176e-07 / (16.0 + y2)
    )
    return Sigma1

@jit
def rewofz(x, y):
    """Real part of wofz (Faddeeva) function based on Algorithm 916.
    We apply a=0.5 for Algorithm 916.
    Args:
        x: x < ncut/2
        y:
    Returns:
        jnp.array: Real(wofz(x+iy))
    """
    xy = x * y
    exx = jnp.exp(-x * x)
    f = exx * (
        erfcx(y) * jnp.cos(2.0 * xy) + x * jnp.sin(xy) / jnp.pi * jnp.sinc(xy / jnp.pi)
    )
    y2 = y * y
    Sigma23 = jnp.sum(
        (jnp.exp(-((an + x) ** 2)) + jnp.exp(-((an - x) ** 2))) / (a2n2 + y2)
    )
    Sigma1 = faddeeva_sigma1(exx, y2)
    f = f + y / jnp.pi * (-jnp.cos(2.0 * xy) * Sigma1 + 0.5 * Sigma23)
    return f

@jit
def asymptotic_wofz_real(x, y):
    """Asymptotic representation of Real(wofz(x+iy))"""
    z = x + y * (1j)
    a = 1.0 / (2.0 * z * z)
    q = (1j) / (z * jnp.sqrt(jnp.pi)) * (1.0 + a * (1.0 + a * (3.0 + a * 15.0)))
    return jnp.real(q)

@jit
def hjert(x, a):
    """Voigt-Hjerting function H(x, a) = Real(wofz(x + ia))
    """
    r2 = x * x + a * a
    # Use rewofz for small arguments, asymptotic for large
    return jnp.where(r2 < 111.0, rewofz(x, a), asymptotic_wofz_real(x, a))
