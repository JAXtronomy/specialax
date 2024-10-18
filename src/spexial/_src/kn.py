import jax
from jax import lax
from jax import numpy as jnp 
from jax.scipy.special import i0, i1
from jax.scipy.special import gamma as jax_gamma

euler_gamma = 0.57721566490153286061

def K0(z): 
    """
    Modified Bessel function of the second kind of order 0. 
    See Zhang and Jin SPECIAL_FUNCTIONS in FORTRAN77 for algorithm. 

    Parameters
    ----------
    z : float
        Input value

    Returns
    -------
    float
        Value of K0(z)
    
    """

    def K0_small(z):

        # n = 30 sufficient for abstol ~ 1e-12 and reltol ~ 1e-8
        int_ary = jnp.arange(1., 31)

        harmonic_ary = jnp.cumsum(1. / jnp.arange(1, 31))

        return -(jnp.log(z/2.) + euler_gamma) * i0(jnp.where(z < 600., z, 600.)) + jnp.sum(
            harmonic_ary * (z/2.)**(2.*int_ary) / jax_gamma(int_ary+1)**2.
        )

    def K0_large(z): 

        # n = 10 sufficient for abstol ~ 1e-12 and reltol ~ 1e-8

        int_ary = jnp.arange(1., 11)

        prod_term = jnp.cumprod(-(2.*int_ary - 1.) / (2.*int_ary) * (2.*int_ary - 1.)**2.) 

        res = 1. / 2. / z / i0(jnp.where(z < 600., z, 600.)) * (1. + jnp.sum((-1)**int_ary * prod_term / (2.*z)**(2.*int_ary)))

        return jnp.where(z < 600, res, 0.)

    return lax.cond(z < 9, K0_small, K0_large, z)

def K1(z): 
    """
    Modified Bessel function of the second kind of order 1. 
    
    Parameters
    ----------
    z : float
        Input value

    Returns
    -------
    float
        Value of K1(z)
    
    """

    def K1_small(z): 

        return (1 / z - i1(z) * K0(z)) / i0(z)

    return jnp.where(z < 600., K1_small(jnp.where(z < 600. , z, 600.)), 0.)


def K2(z): 
    """
    Modified Bessel function of the second kind of order 2. 
    
    Parameters
    ----------
    z : float
        Input value

    Returns
    -------
    float
        Value of K2(z)
    
    """

    return K0(z) + 2 / z * K1(z) 