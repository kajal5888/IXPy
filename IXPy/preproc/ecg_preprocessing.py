
# from IXPy.utils import utils
import numpy as np
__all__ = ["rmssd", "RRIndex2MilliSec"]


def rmssd(HBPM):
    """
    This function estimates the Root Mean Square of Successive Differences (RMSSD).
    Args:
        HBPM (list or np.array): A list or array of heartbeats per minute (BPM) values.
    Returns:
        float: The RMSSD value.
    """

    # Convert HBPM to RR intervals (in milliseconds)
    rr_intervals = 60000 / np.array(HBPM)  # RR intervals in ms

    # Compute successive differences
    rr_diff = np.diff(rr_intervals)

    # Compute RMSSD
    rmssd = np.sqrt(np.mean(rr_diff ** 2))

    return rmssd


def RRIndex2MilliSec(R_peaks_index, estimator='median', sampling_freq=256, min_rr=250, max_rr=2000, interp=True):
    """
    Convert R-peak indices or timestamps to RR intervals in milliseconds.

    - Accepts sample indices, timestamps in seconds or timestamps in milliseconds (auto-detected).
    - Returns successive RR intervals in ms.
    - Filters implausible RR values outside [min_rr, max_rr] ms and optionally interpolates isolated artifacts.

    Parameters:
        R_peak_index (array-like): R-peak sample indices or timestamps.
        sampling_freq (float): Sampling frequency (Hz) used when input are indices (default 256).
        min_rr (float): Minimum plausible RR in ms (default 250 ms).
        max_rr (float): Maximum plausible RR in ms (default 2000 ms).
        interp (bool): If True, interpolate isolated invalid RR values; otherwise leave as NaN.

    Returns:
        np.ndarray: RR intervals in milliseconds (float64). Empty array if fewer than 2 peaks.
    """

    r = np.asarray(R_peaks_index, dtype=float)
    if r.size < 2:
        return np.empty(0, dtype=float)

    diff_r = np.diff(r)
    if estimator == 'median':
        med_diff = np.median(np.abs(diff_r))
    elif estimator == 'mean':
        med_diff = np.mean(np.abs(diff_r))

    # Heuristic auto-detect input unit based on typical spacing
    # - median diff >> 1000 -> values are ms timestamps
    # - otherwise assume sample indices
    if med_diff > 1000:
        rr_ms = diff_r.copy()                      # already in ms
    else:
        rr_ms = (diff_r / float(sampling_freq)) * 1000.0  # indices -> ms

    rr_ms = rr_ms.astype(float)

    # Mark implausible intervals
    bad_mask = (rr_ms < min_rr) | (rr_ms > max_rr)

    if interp and bad_mask.any():
        x = np.arange(rr_ms.size)
        good = ~bad_mask
        # Interpolate only if there are at least two good points
        if good.sum() >= 2:
            rr_ms[bad_mask] = np.interp(x[bad_mask], x[good], rr_ms[good])
        else:
            rr_ms[bad_mask] = np.nan
    return rr_ms
