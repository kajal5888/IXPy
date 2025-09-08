# from IXPy.scripts.utils import utils
# from IXPy.scripts.io import load_EEG

# from .utils.utils import ReadCSV
# from .io.load_EEG import load_EEG

from .utils import *
from .io import *
from .preproc import *

# # print(help(ReadCSV))
# # Manage user-exposed namespace imports
# # __all__ = ["ReadCSV", "load_EEG"]
__all__ = []
__all__.extend(utils.__all__)
__all__.extend(io.__all__)
__all__.extend(preproc.__all__)
# from IXPy.io.load_EEG import Load_Data
# from IXPy.utils.utils import DJ_Print

# __all__ = ["Load_Data", "DJ_Print"]
