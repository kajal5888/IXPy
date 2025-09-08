#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from IXPy.utils import utils

__all__ = ["Load_Data",]


def Load_Data(Filename, Preload, Device="Brainvision"):
    """_summary_

    Args:
        Filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    import mne
    if utils._dir_and_file_check(Filename):
        Task = "" if Preload else "Meta "
        utils.DJ_Print(f"Loading {Task}Data: {os.path.basename(Filename)}")
        if Device == "Brainvision":
            Data = mne.io.read_raw_brainvision(Filename, preload=Preload)
        elif Device == 'Somnomedics':
            Data = mne.io.read_raw_edf(Filename, preload=Preload)
        else:
            utils.DJ_Print(
                f"Target File : {os.path.basename(Filename)} MISSING", ColType='warning')
    return Data
