import os
from IXPy.utils import utils
import mne
import matplotlib.pyplot as plt


__all__ = ["Filter_EEG", "Set_Montage", "PSD", "Spectrogram"]


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


def Spectrogram(raw, subject_id, channel_names, fmin, fmax, tmin, tmax, fs, nperseg, noverlap, save_full_path, save, normalize, overwrite, tfr_method, version_type, show):
    """
    Compute and save spectrograms for specified EEG channels using Short-Time Fourier Transform (STFT).

    Args:
        raw (mne.io.Raw): The MNE Raw object containing EEG data.
        channel_names (list of str): List of channel names for which spectrograms will be generated.
        fmin (float): Minimum frequency (Hz) for the spectrogram.
        fmax (float): Maximum frequency (Hz) for the spectrogram.
        fs (float): Sampling frequency (Hz) of the EEG data.
        nperseg (int): Number of samples per segment for STFT.
        noverlap (int): Number of overlapping samples between segments.
        save_full_path (str): Directory path where the spectrogram images will be saved.
        save (bool, optional): Whether to save the spectrogram plots. Defaults to True.
        normalize (bool, optional): Whether to normalize the power values before visualization. Defaults to True.

    Returns:
        None

    Notes:
        - Uses `compute_spectrogram()` to calculate power spectral density.
        - Normalizes power values using `normalize_data()` if `normalize=True`.
        - Saves the spectrogram plots in the `save_full_path + '/Spectrogram/'` directory if `save=True`.
        - Uses a logarithmic color scale (`10 * np.log10(power)`) for better visualization of power levels.
    """
    if utils.DirCheck(save_full_path, overwrite):
        import numpy as np
        import matplotlib.pyplot as plt
        for ch in channel_names[-1:]:
            utils.DJ_Print(
                f"Generating Spectrogram for Subject: {subject_id}, channel: {ch}")
            TFR_Filename = f"{save_full_path}{version_type}_{tfr_method}_{ch}.png"
            fmin, fmax, fs = fmin, fmax, raw.info['sfreq']
            frequencies = np.arange(fmin, fmax, .1)
            n_cycles = 8

            if version_type == 'SonyDAB':
                from scipy.signal import spectrogram
                data, times = raw.copy().get_data(
                    picks=[ch], return_times=True)
                # Get the 1D signal of the chosen channel
                data = data[0]

                # Perform STFT to get spectrogram
                frequencies, times_stft, Zxx = spectrogram(
                    data, fs=fs, nperseg=nperseg, noverlap=noverlap)
                freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
                frequencies = frequencies[freq_mask]
                power = Zxx[freq_mask, :]
                if normalize:
                    power = (power - np.min(power)) / \
                        (np.max(power) - np.min(power))

                vmin, vmax = -60, -10  # -140, -20

                plt.figure(figsize=(10, 6))
                plt.pcolormesh(times_stft, frequencies, 10 * np.log10(power),
                               shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
                plt.colorbar(label='Power (dB)')
                plt.title(
                    f' {subject_id}: Time-Frequency Representation (STFT) for {ch}')
                plt.xlabel('Time (s)')
                plt.tight_layout()
                # plt.show()

                if save:
                    plt.savefig(TFR_Filename, format='.png')
            elif version_type == 'MNE_Python':

                # Create a fake event spanning the entire raw
                event_id = 1
                events = np.array([[0, 0, event_id]])
                tmin, tmax = 0, raw.times[-1]

                # One long "epoch" only with target channels
                Data_Epoched = mne.Epochs(raw.copy(), events, event_id=event_id, tmin=tmin, tmax=tmax,
                                          baseline=None, preload=True, picks=ch)

                tfr = utils.TFR_MNE(
                    Filename=TFR_Filename.replace(".png", ""))
                if tfr_method == 'Morlet':
                    if os.path.isfile(tfr.Filename):
                        DataTFR = tfr.Load()
                    else:
                        # DataTFR = mne.time_frequency.tfr_morlet(Data_Epoched, freqs=frequencies, n_cycles=n_cycles,
                        #                                         use_fft=True, return_itc=False, decim=5, n_jobs=1)
                        DataTFR = Data_Epoched.compute_tfr(method="morlet", freqs=frequencies, n_cycles=n_cycles,
                                                           use_fft=True, return_itc=False, decim=5, n_jobs=1)
                        tfr.Save(DataTFR)
                elif tfr_method == 'MultiTaper':
                    if os.path.isfile(f"{tfr.Filename}-tfr.h5"):
                        DataTFR = tfr.Load()
                    else:
                        DataTFR = mne.time_frequency.tfr_multitaper(Data_Epoched, freqs=frequencies, time_bandwidth=4, n_cycles=n_cycles,
                                                                    return_itc=False, decim=5, n_jobs=1)
                        tfr.Save(DataTFR)
                else:
                    utils.DJ_Print(
                        "Requested TFR method is not implemented", "warning")

                vmin, vmax,  Yscale, cb, CMAP = (
                    DataTFR.data.min()*.5, DataTFR.data.max()*.2,  'linear', False, plt.cm.jet)
                plt.close('all')
                plt.figure(figsize=(10, 8))
                Ax = plt.axes()
                DataTFR.plot(picks=ch, colorbar=cb, yscale=Yscale,
                             cmap=CMAP, axes=Ax, show=False)
                if save:
                    plt.savefig(TFR_Filename, format='png')
                if show:
                    plt.show(block=False)
