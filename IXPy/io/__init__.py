import inspect
from . import load_EEG, load_Psychopy

__all__ = []
for module in [load_EEG, load_Psychopy]:
    funcs = [name for name, obj in inspect.getmembers(
        module, inspect.isfunction) if obj.__module__ == module.__name__]
    __all__.extend(funcs)
    for name in funcs:
        globals()[name] = getattr(module, name)
