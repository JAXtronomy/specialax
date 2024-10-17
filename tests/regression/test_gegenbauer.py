"""Test `spexial.gegenbauer` matches `scipy.spexial`."""

import numpy as np
from hypothesis import example, given, settings, strategies as st
from scipy.special import eval_gegenbauer as scipy_eval_gegenbauer

from spexial import eval_gegenbauer


@given(
    n=st.integers(min_value=0, max_value=20),
    alpha=st.floats(min_value=-0.49, max_value=10.0),  # TODO: 0.5
    x=st.floats(min_value=-1.0, max_value=1.0),
)
@example(n=2, alpha=1.0, x=1.0)
@example(n=1, alpha=1.0, x=1.0)
@example(n=0, alpha=1.0, x=1.0)
@settings(deadline=1000)  # Set the maximum time for this test to 1000ms
def test_eval_gegenbauer(n, alpha, x):
    """Test `spexial.gegenbauer` matches `scipy.spexial`."""
    result = eval_gegenbauer(n, alpha, x)
    expected = scipy_eval_gegenbauer(n, alpha, x)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)
