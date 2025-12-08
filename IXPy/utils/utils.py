#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Dr. Diljit Singh Kajal
Title: SonyDAB - Behaviour Analysis 20224
Company: Institut für experimentelle Psychophysiologie GmbH
Date : 28 May 2025
email: d.kajal@ixp-duesseldorf.de

"""

from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import pyedflib
from pathlib import Path
import os
import mne
import socket

__all__ = [
    "DJ_StructureTemplate",
    "PrintColors",
    "DJ_Print",
    "ReadCSV_MaxHealthBand",
    "Preprocessing_MaxHealthBand",
    "ReadCSV",
    "ReadEDF",
    "load_json",
    "_dir_and_file_check",
    "FolderCheck",
    "DirCheck",
    "TFR_MNE",
    "Sony_SubjectList",
    "Project_year",
    "Workpackages"
]


class DJ_StructureTemplate(dict):

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __repr__(self):
        if len(self.__dict__.items()) < 1:
            return '{}'

        parts = []
        for k in self.__dict__.keys():
            if isinstance(self.__dict__[k], DJ_StructureTemplate):
                rrr = repr(self.__dict__[k])
                parts.append(k + "\t[struct]")
            else:
                parts.append(k + ":\t" + repr(self.__dict__[k]))
        result = '\n'.join(parts)
        return result

    def __bool__(self):
        return len(self.__dict__.keys()) > 0

    def __getitem__(self, key):
        val = getattr(self, key)
        return val

    def __setitem__(self, key, val):
        return setattr(self, key, val)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


class PrintColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def DJ_Print(STR, ColType='green'):
    ''' Available colors are : header, blue, green, cyan, warning, fail, bold and underline'''
    if ColType.lower() == 'header':
        print(PrintColors.HEADER + STR + PrintColors.ENDC)
    elif ColType.lower() == 'blue':
        print(PrintColors.BLUE + STR + PrintColors.ENDC)
    elif ColType.lower() == 'cyan':
        print(PrintColors.CYAN + STR + PrintColors.ENDC)
    elif ColType.lower() == 'green':
        print(PrintColors.GREEN + STR + PrintColors.ENDC)
    elif ColType.lower() == 'warning':
        print(PrintColors.WARNING + STR + PrintColors.ENDC)
    elif ColType.lower() == 'fail':
        print(PrintColors.FAIL + STR + PrintColors.ENDC)
    elif ColType.lower() == 'underline':
        print(PrintColors.UNDERLINE + STR + PrintColors.ENDC)
    elif ColType.lower() == 'bold':
        print(PrintColors.BOLD + STR + PrintColors.ENDC)


def DirCheck(FileName, Replace):
    """This python definition check if the target file or directory exists or not.
    If the Replace is true then it returns True of replace the target directory


    Args:
        FileName (str): Name of the file
        Replace (bool): True or False

    Returns: Boolean True or False
    """
    if Replace:
        return True
    Check = Path(FileName)
    if Check.exists():
        if Check.is_dir():
            return False if os.path.isdir(FileName) else True
        if Check.is_file():
            return False if os.path.isfile(FileName) else True
    else:
        return True


def ReadCSV_MaxHealthBand(filename):
    """_summary_

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = pd.read_csv(filename)
    df['Time'] = pd.to_datetime(
        df['timestamp'], unit='ms').dt.strftime('%H:%M:%S')

    print(f"Experiment started at {df['Time'][0]}")
    print(f"Experiment Ended at {list(df['Time'])[-1]}")
    print(f"Sampling Rate is {1000/np.mean(np.diff(df['timestamp']))}")
    return df


def Preprocessing_MaxHealthBand(df, ppg_columns):
    """_summary_

    Args:
        df (_type_): _description_
        ppg_columns (_type_): _description_

    Returns:
        _type_: _description_
    """
    for col in (ppg_columns):
        # Convert to numeric and handle non-numeric values
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Replace zero values with NaN
        df.loc[df[col] == 0, col] = pd.NA

    df[ppg_columns] = df[ppg_columns].interpolate(method='linear')

    # Apply Savitzky-Golay filter for smoothing
    for col in ppg_columns:
        df[col] = savgol_filter(df[col].fillna(
            0).values, window_length=51, polyorder=3)
    return df


class ReadCSV:
    from pathlib import Path

    def __init__(self, filename, device, encoding='utf-8', pandas_kwargs=None):
        self.filename = Path(filename)
        self.device = device.lower()
        self.encoding = encoding
        self.pandas_kwargs = pandas_kwargs or {}

    def __repr__(self) -> str:
        return f"ReadCSV(filename={str(self.filename)!r}, device={self.device!r})"

    def _validate(self) -> None:
        if not self.filename.exists():
            raise FileNotFoundError(f"CSV file not found: {self.filename}")

    def Read(self):
        """This function reads the CSV file
        """
        self._validate()
        try:
            if self.device == 'shimmer':
                header_idx = 0
                dF = pd.read_csv(self.filename, sep=',',
                                 low_memory=False, encoding=self.encoding, **self.pandas_kwargs)
                dF = dF.reset_index()
                dF.columns = dF.iloc[header_idx]
                dF.drop(index=[header_idx, header_idx + 1], axis=0,
                        inplace=True)
            else:
                dF = pd.read_csv(
                    self.filename, encoding=self.encoding, **self.pandas_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read CSV {self.filename}: {exc}") from exc
        return dF


def ReadEDF(filename):
    f = pyedflib.EdfReader(filename)
    if 'Somno' in f.getEquipment():
        Device = f.getEquipment
    else:
        Device = 'Unknown'

    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sampling_rate = f.getSampleFrequency()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)

    return sigbufs


def load_json(filename):
    import json
    with open(filename, 'r') as f:
        info = json.load(f)
    return info


def _dir_and_file_check(FileName):
    """This python definition check if the target file or directory exists or not.
    If the Replace is true then it returns True of replace the target directory


    Args:
        FileName (str): Name of the file
        Replace (bool): True or False

    Returns: Boolean True or False
    """

    Check = Path(FileName)
    if Check.exists():
        if Check.is_dir():
            return True if os.path.isdir(FileName) else False
        if Check.is_file():
            return True if os.path.isfile(FileName) else False
    else:
        return True


def FolderCheck(Path, MakeFolder=True):
    if not os.path.isdir(Path) and MakeFolder:
        DJ_Print('Creating Folder...', 'warning')
        os.mkdir(Path)


class TFR_MNE():
    def __init__(self, Filename):
        self.Filename = Filename
        """ Insert the File checking procedure"""

    def Save(self, File):
        mne.time_frequency.write_tfrs(
            fname=self.Filename + '-tfr.h5', tfr=File, overwrite=True)

    def Load(self):
        loadedTFR = mne.time_frequency.read_tfrs(
            fname=self.Filename + '-tfr.h5')
        return loadedTFR


def Workpackages():
    return {'WP2': 'WP2/Data Sets_MainExperimentRawData/',
            'WP1': 'WP1/Data Sets_MainExperimentRawData/', }


def Project_year(Project_Name):
    year_project = {'DAB': '2024_DAB', 'MAS': '2025_MAS'}
    assert Project_Name in year_project, DJ_Print(
        "Please enter the correct Project Name")
    return year_project[Project_Name]


class Sony_SubjectList():
    """Project Name = "DAB", "MAS"
    """

    def __init__(self, Project_Name="DAB"):
        self.project_name = Project_year(Project_Name)

    def SonyDAB_SubjectList(self, Workpackage):
        if 'IXP' in socket.gethostname():
            DJ_Print('Please enter the BasePath', 'warning')
            pass
            # BasePath = ""
        elif "dkajal" in socket.gethostname():
            BasePath = f"/Users/dkajal/OneDrive - Institut für experimentelle Psychophysiologie GmbH/Projekte_laufend/{self.project_name}_Sony/"
        DataPath = f"{BasePath}{Workpackages()[Workpackage]}".replace(
            '\\', '/')
        SubjectFolders = [Subject for Subject in os.listdir(
            DataPath) if 'WP' in Subject and '.xlsx' not in Subject]
        SubjectFolders.sort()
        return SubjectFolders


# def DJ_MNE_formatData(DataToBeConverted, WindowLength, ReduceTrials=True, ReduceTrialNum=DJ_AnalysisParameters().ReduceTrialNum):
#     WL = _WindowLength(WindowLength)
#     ch_types, ch_names, LowPass, HighPass = list(), list(), 0.1, 300
#     _ = [ch_types.append('eeg')
#          for ct in range(0, len(DataToBeConverted.channel))]
#     _ = [ch_names.append(x) for x in DataToBeConverted.channel]
#     sfreq = DataToBeConverted.samplerate
#     ch_names.append('TriggerChannel')
#     ch_types.append('stim')
#     info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
#     Data_Con = np.concatenate((DataToBeConverted.data[:].T, np.zeros(
#         (1, DataToBeConverted.data.shape[0]))), axis=0)
#     RawMNEData = mne.io.RawArray(Data_Con, info=info, verbose=0)
#     WindowInSamples = int(np.ceil(DataToBeConverted.samplerate * WL))
#     if ReduceTrials:
#         Array = np.linspace(
#             0, DataToBeConverted.sampleinfo[-1, 1], ReduceTrialNum + 1, dtype=int)
#         events = np.ones((ReduceTrialNum, 3), dtype=int)
#         events[:, 0] = Array[:-1]
#         events[:, 1] = Array[1:] - 1
#         events[:, 2] = np.arange(ReduceTrialNum) + 1
#     else:
#         events = np.ones((len(DataToBeConverted.sampleinfo[:, 0]), 3))
#         events[:, 0] = DataToBeConverted.sampleinfo[:, 0]
#         events[:, 1] = DataToBeConverted.sampleinfo[:, 1]
#     # events[:, 2] = np.arange(len(DataToBeConverted.sampleinfo[:, 1]))
#     RawMNEData.add_events(
#         events=events, stim_channel='TriggerChannel', replace=True)
#     if not RawMNEData.preload:
#         RawMNEData.load_data()
#     RawMNEData.filter(LowPass, HighPass)
#     Events = mne.find_events(
#         RawMNEData, stim_channel='TriggerChannel', initial_event=True)
#     EpochedData = mne.Epochs(RawMNEData, events, tmin=0, tmax=np.mean(np.diff(
#         events[:, :2]) / sfreq), baseline=None, reject=None, reject_by_annotation=None)
#     return RawMNEData, EpochedData
