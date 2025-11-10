import inspect
from . import load_EEG, load_Psychopy, load_EmbracePlus

__all__ = []
for module in [load_EEG, load_Psychopy, load_EmbracePlus]:
    funcs = [name for name, obj in inspect.getmembers(
        module, lambda o: inspect.isfunction(o) or inspect.isclass(o)) if getattr(obj, "__module__", None) == module.__name__]
    __all__.extend(funcs)
    for name in funcs:
        globals()[name] = getattr(module, name)
