"""
Translation of the cython implementation of complex Spence
from scipy to work with jax.

Adapted from https://github.com/scipy/scipy/blob/8971d5e9b72931987b7d3c5a25da1a8e7e5485d0/scipy/special/_spence.pxd
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

MAXITER = 500


def complex_spence_series_0(z):
    """
    Small z branch, uses a series expansion about :math:`z = 0`
    for :math:`|z| < 0.5`.
    """
    
    nn = jnp.arange(1, MAXITER)
    temp = z**nn / nn
    sum1 = jnp.sum(temp / nn)
    sum2 = jnp.sum(temp)
    
    return np.pi**2 / 6 - sum1 + jnp.log(z) * sum2


def complex_spence_series_1(z):
    """
    Middle branch, an expansion around :math:`z = 1`
    for :math:`|z| > 0.5` and :math:`|1 - z| > 1`.
    """
    z = 1 - z
    nn = jnp.arange(1, MAXITER)
    res = jnp.sum(z**nn / (nn * (nn + 1) * (nn + 2))**2)

    res *= 4 * z**2
    res += 4 * z + 5.75 * z**2 + 3 * (1 - z**2) * jnp.log1p(-z)
    res /= 1 + 4 * z + z**2
    return res


def complex_spence_series_2(z):
    """
    Large :math:`z` branch for :math:`|z| > 0.5` and
    :math:`|1 - z| > 1`. Uses a reflection expression.
    """
    return (
        -complex_spence_series_1(z / (z - 1))
        - np.pi**2 / 6
        - jnp.log(z - 1)**2 / 2
    )


@jax.jit
def spence(z):
    """
    Return the Spence dilogarithm for complex input using
    branches dependent on the value of the input.
    """
    return jax.lax.select(
        abs(z) < 0.5,
        complex_spence_series_0(z),
        jax.lax.select(
            abs(1 - z) > 1,
            complex_spence_series_2(z),
            complex_spence_series_1(z),
        ),
    )


def spence_jvp(primals, tangents):
    """Template of spence jvp that different implementations can use"""
    (z,) = primals
    (z_dot,) = tangents
    dspence = jnp.log(z) / (1 - z)
    return spence(z), z_dot * dspence


spence = jax.custom_jvp(spence)
spence.defjvp(spence_jvp)
