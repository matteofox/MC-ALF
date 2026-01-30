try:
    from .._version import __version__
except(ImportError):
    pass

from . import hires_fitter

__all__ = ['hires_fitter']
