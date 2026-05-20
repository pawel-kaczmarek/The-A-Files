from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("the-a-files")
except PackageNotFoundError:  # package not installed (running from source tree)
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
