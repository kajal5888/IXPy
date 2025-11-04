
import inspect
from . import preprocessing
from . import ecg_preprocessing


def _public_functions(module):
    return [name for name, obj in inspect.getmembers(module, inspect.isfunction)
            if obj.__module__ == module.__name__]


_preproc_names = _public_functions(preprocessing)
_ecg_names = _public_functions(ecg_preprocessing)

seen = set()
__all__ = []
for name in _preproc_names + _ecg_names:
    if name not in seen:
        seen.add(name)
        __all__.append(name)

for name in _preproc_names:
    globals()[name] = getattr(preprocessing, name)
for name in _ecg_names:
    globals()[name] = getattr(ecg_preprocessing, name)
