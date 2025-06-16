#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Dr. Diljit Singh Kajal
Title: SonyDAB - Behaviour Analysis 20224
Company: Institut für experimentelle Psychophysiologie GmbH
Date : 28 May 2025
email: d.kajal@ixp-duesseldorf.de

"""
import os
import pandas as pd
from IXPy.utils import utils

__all__ = ["Load_Psychopy",]


def Load_Psychopy(Filename):
    """_summary_

    Args:
        Filename (_type_): _description_

    Returns:
        _type_: _description_
    """

    if utils._dir_and_file_check(Filename):
        Task = ""
        utils.DJ_Print(f"Loading {Task}Data: {os.path.basename(Filename)}")
        Data = pd.read_csv(Filename)
    else:
        utils.DJ_Print(
            f"Target File : {os.path.basename(Filename)} MISSING", ColType='warning')
    return Data
