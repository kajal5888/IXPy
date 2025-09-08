import os
from IXPy.utils import utils
import mne
import matplotlib.pyplot as plt

__all__ = ["Filter_EEG", "Set_Montage", "PSD"]


def Filter_EEG(raw, notch=True, notch_freqs=(50, 100, 150), band_pass=True, l_freq=1.0, h_freq=None, fir_design='firwin'):
    """
    Apply filtering to EEG data.

    Args:
        raw (mne.io.Raw): The MNE Raw object containing EEG data.
        notch (str, optional):
            - 'True' to apply a notch filter to remove line noise.
            - 'False' (default) to skip notch filtering.
        notch_freqs (float or list, optional): The frequency/frequencies to remove using the notch filter.
            - Default is 50 Hz (common power line noise frequency).
        band_pass (str, optional):
            - 'True' to apply a band-pass filter.
            - 'False' (default) to skip band-pass filtering.
        l_freq (float, optional): The lower cutoff frequency for the band-pass filter. Default is 1.0 Hz.
        h_freq (float or None, optional): The upper cutoff frequency for the band-pass filter.
            - Default is None (no high cutoff, allowing a high-pass filter instead).
        fir_design (str, optional): FIR filter design method. Default is 'firwin'.

    Returns:
        mne.io.Raw: The filtered EEG data.

    Notes:
        - The function applies a band-pass filter if `band_pass='True'`.
        - A notch filter is applied to remove power line noise if `notch='True'`.
        - Filtering is performed in-place on the `raw` object.
    """
    # Filter all channels
    if band_pass:
        print('-----band pass filteration in process-----')
        raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design=fir_design)
    else:
        print('band pass filteration not applied')

    if notch:
        print('-----removing line noise using notch filter-----')
        raw.notch_filter(freqs=notch_freqs)
    else:
        print('line noise is not removed')

    return raw


def Set_Montage(raw, mapping, montage):
    """
    Set the EEG channel montage and rename channels to match the standard 10-20 system.

    Args:
        raw (mne.io.Raw): The MNE Raw object containing EEG data.
        mapping (dict): A dictionary mapping old channel names to new names (e.g., {'FpZ': 'Fpz', 'OZ': 'Oz'}).
        montage (str): The name of the standard montage to apply (e.g., 'standard_1020').

    Returns:
        mne.io.Raw: The EEG data with updated channel names and applied montage.

    Notes:
        - The function renames channels based on the provided `mapping` dictionary.
        - It applies a standard 10-20 EEG montage using `mne.channels.make_standard_montage()`.
        - If some channels are missing from the montage, they are ignored.
    """

    # changing names to match the standard 1020 nomenclature
    raw.rename_channels(mapping)
    # making standard montage using 10-20 system
    ten_twenty_montage = mne.channels.make_standard_montage(montage)
    # applying the 10-20 montage to raw data
    raw.set_montage(ten_twenty_montage, on_missing="ignore")

    return raw


def PSD(raw, subject_id, channel=None, fmin=1, fmax=100, tmin=None, tmax=None,
        n_fft=1000, n_per_seg=1000, n_overlap=0, save=False, save_full_path=None,  plotting=True, show=False, Overwrite=False):
    """
    Compute and plot the Power Spectral Density (PSD) of EEG data.

    Args:
        raw (mne.io.Raw): The MNE Raw object containing EEG data.
        average (None, optional): Unused parameter; can be removed or implemented if needed.
        channel (list of str or str, optional):
            - Specific channel(s) to compute PSD for.
            - If None (default), all channels are used.
        fmin (float, optional): Minimum frequency (Hz) for PSD computation. Default is 1 Hz.
        fmax (float, optional): Maximum frequency (Hz) for PSD computation. Default is 100 Hz.
        n_fft (int, optional): Number of FFT points. Default is 1000.
        n_per_seg (int, optional): Length of each segment for Welch’s method. Default is 1000.
        n_overlap (int, optional): Number of points to overlap between segments. Default is 0.
        save_full_path (str, optional): Directory path to save the PSD plot. Required if `save=True`.
        save (bool, optional): Whether to save the PSD plot. Default is True.

    Returns:
        None

    Notes:
        - If `channel` is specified, only the selected channels are used for PSD computation.
        - The PSD is computed using `raw.compute_psd(method="auto")`.
        - The function plots the PSD for each EEG channel on a semilogarithmic scale.
        - If `save=True`, the plot is saved in the specified `save_full_path` under a `PSD/` subdirectory.
    """

    if utils.DirCheck(save_full_path, Overwrite):
        raw = raw if channel is None else raw.copy().pick_channels(channel)

        spectrum = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
                                   n_fft=n_fft, n_per_seg=n_per_seg, n_overlap=n_overlap)
        if plotting:
            plt.figure(figsize=(10, 8))
            Ax = plt.axes()
            spectrum.plot(axes=Ax, spatial_colors=True, show=False)
            Ax.set(title=subject_id, xlabel='Frequency (Hz)',
                   ylabel=r'Power spectral density ($\mu V^2$/Hz)(dB)')
            _ = [line.set_linewidth(1.5) for line in Ax.get_lines()]
            Ax.set_ylim(-10, 50)
            if save:
                plt.savefig(save_full_path, format='png')
            if show:
                plt.show(block=False)
