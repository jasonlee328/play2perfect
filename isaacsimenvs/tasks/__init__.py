"""Task registry for isaacsimenvs.

Each task subpackage registers itself with gymnasium on import (side effect
in its ``__init__.py``). Importing ``isaacsimenvs`` (or any child) is enough
to expose all task ids to ``gym.make`` / ``gym.spec``.
"""

from . import play              # gym.register("Isaacsimenvs-Play-Direct-v0", ...)
from . import precise_assembly  # gym.register("Isaacsimenvs-PreciseAssembly-Direct-v0", ...)

__all__ = ["play", "precise_assembly"]
