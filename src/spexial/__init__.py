"""Special functions."""

__all__ = [
    # Gegenbauer
    "eval_gegenbauer",
    "eval_gegenbauers",  # NOTE: not in scipy.special
]

from ._src.gegenbauer import eval_gegenbauer, eval_gegenbauers
from ._version import version as __version__  # noqa: F401
