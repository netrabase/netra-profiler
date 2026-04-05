from importlib import metadata

# Single Source of Truth for Versioning
try:
    __version__ = metadata.version("netra-profiler")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev"

# This allows users to do: "from netra_profiler import Profiler"
# instead of "from netra_profiler.core import Profiler"
from .profiler import Profiler

__all__ = ["Profiler", "__version__"]
