"""Jax implementation of the Gegenbauer polynomials."""

import jax.numpy as jnp
from functools import partial
import jax


@jax.jit
def C0(x):
    """
    Initialize with C0
    """
    return 1.0


@jax.jit
def C1(x, alpha):
    """
    Initialize with C1
    """
    return 2 * alpha * x


@jax.jit
def C_n_plus_1(carry, n):
    """
    Function to be used in scan
    """
    x, alpha, Cn, Cn_minus_1 = carry
    accumulate = (2 * (n + alpha) * x * Cn - (n + 2 * alpha - 1) * Cn_minus_1) / (n + 1)
    carryover = (x, alpha, accumulate, Cn)

    return carryover, accumulate


@partial(jax.jit, static_argnums=(2,))
def gegenbauer_polynomials(x=None, alpha=None, N=None):
    """
    Implements scan of C_n_plus_1
    Returns every order of the polynomial computed along the way
    """
    C0_val = C0(x)
    C1_val = C1(x, alpha)

    carry = (x, alpha, C1_val, C0_val)
    n_values = jnp.arange(
        1, N
    )  # n starts from 1 since we already initialized 0->1 above

    _, C_values = jax.lax.scan(C_n_plus_1, carry, n_values)

    return jnp.hstack([C0_val, C1_val, C_values])


@partial(jax.jit, static_argnums=(2,))
def C_n_alpha(x=None, alpha=None, N=None):
    """
    Vmapped version of gegenbauer_polynomials,
    assumes batched x inputs
    """
    return jax.vmap(gegenbauer_polynomials, in_axes=(0, None, None))(x, alpha, N)[:, -1]