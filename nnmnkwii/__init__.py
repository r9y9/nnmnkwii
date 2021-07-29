"""Library to build speech synthesis systems designed for prototyping.

https://github.com/r9y9/nnmnkwii
"""

try:
    from .version import __version__  # NOQA
except ImportError:
    raise ImportError("BUG: version.py doesn't exist. Please file a bug report.")
