"""Special functions."""

__all__ = ["eval_gegenbauer", "eval_gegenbauers"]

from ._src.gegenbauer import eval_gegenbauer, eval_gegenbauers
from ._version import version as __version__  # noqa: F401
