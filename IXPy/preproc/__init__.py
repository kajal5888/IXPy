
import inspect
from . import preprocessing

__all__ = [name for name, obj in inspect.getmembers(
    preprocessing, inspect.isfunction) if obj.__module__ == preprocessing.__name__]

for name in __all__:
    globals()[name] = getattr(preprocessing, name)
