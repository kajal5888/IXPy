#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Dr. Diljit Singh Kajal
Title: SonyDAB - Behaviour Analysis 20224
Company: Institut für experimentelle Psychophysiologie GmbH
Date : 28 May 2025
email: d.kajal@ixp-duesseldorf.de

"""

# from load_EEG import *
# import load_EEG

# print(dir(load_EEG))
# from . import (
#     load_EEG,
# )

from .load_EEG import Load_Data
from .load_Psychopy import Load_Psychopy
__all__ = ["Load_Data", "Load_Psychopy"]
# __all__.extend(load_EEG.__all__)
