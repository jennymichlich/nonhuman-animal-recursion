"""
Call-onset detection for the IOI/isochrony recursion pipeline.

Approach: bandpass-filter to a species-appropriate frequency band, compute a
smoothed RMS energy envelope, threshold via a robust (median + k*MAD) noise
floor estimate, merge close detections, and drop sub-minimum-duration blips.

This is deliberately simple and auditable (as opposed to a black-box ML
detector) so that every call event can be traced back to a visible spectrogram
feature during QC. Calibrated against manual visual inspection of spectrograms
for at least one exemplar file per species before batch use (see
calibration/ pngs).
"""
import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass(y, sr, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, y)


def energy_envelope_db(y, sr, frame_length=1024, hop_length=256):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    return times, rms_db


def otsu_threshold(values, n_bins=256):
    """
    Classic Otsu's method: split a (possibly bimodal) distribution into two
    classes by maximizing between-class variance. Unlike a median+MAD noise
    floor, this does not assume calls are a minority of frames, so it holds
    up on dense call bouts where vocalization occupies most of the recording.
    """
    hist, edges = np.histogram(values, bins=n_bins)
    hist = hist.astype(float)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return np.median(values)
    sum_total = np.sum(hist * bin_centers)
    sum_bg, w_bg, best_thresh, best_var = 0.0, 0.0, bin_centers[0], -1.0
    for i in range(n_bins):
        w_bg += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += hist[i] * bin_centers[i]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_total - sum_bg) / w_fg
        between_var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if between_var > best_var:
            best_var = between_var
            best_thresh = bin_centers[i]
    return best_thresh


def windowed_otsu_threshold(times, rms_db, window_sec=5.0, hop_sec=1.0,
                             min_dynamic_range_db=8.0):
    """
    Local (sliding-window) Otsu threshold, for recordings where call/background
    SNR drifts over the course of the file (e.g. a bird moving relative to the
    microphone). A single global Otsu threshold silently drops genuine calls
    in quieter stretches; this recomputes the threshold in each window and
    interpolates between window centers to get a per-frame threshold curve.

    Safeguard: a window with no real calls has a unimodal (noise-only)
    distribution, so Otsu still returns *some* split, but that split sits
    right in the middle of the noise floor with very little separation
    between "above" and "below" - meaning that in a genuinely silent stretch,
    the local threshold would otherwise drop until it starts firing on noise
    fluctuations. We detect this by requiring a minimum dynamic range
    (max - min rms_db) within the window; windows below that range fall back
    to the file's global Otsu threshold instead of a local one.
    """
    duration = times[-1]
    centers = np.arange(0, duration + hop_sec, hop_sec)
    global_thresh = otsu_threshold(rms_db)
    local_threshs = []
    for c in centers:
        mask = (times >= c - window_sec / 2) & (times < c + window_sec / 2)
        if mask.sum() < 10:
            local_threshs.append(np.nan)
            continue
        window_vals = rms_db[mask]
        dyn_range = window_vals.max() - window_vals.min()
        if dyn_range < min_dynamic_range_db:
            # no separable call/noise split in this window - don't get
            # aggressive here, defer to the global threshold
            local_threshs.append(global_thresh)
        else:
            local_threshs.append(otsu_threshold(window_vals))
    local_threshs = np.array(local_threshs)
    valid = ~np.isnan(local_threshs)
    if valid.sum() == 0:
        return np.full_like(times, global_thresh)
    local_threshs[~valid] = np.interp(centers[~valid], centers[valid], local_threshs[valid])
    per_frame_thresh = np.interp(times, centers, local_threshs)
    return per_frame_thresh


def detect_calls(y, sr, lo, hi, k=3.0, min_dur=0.03, merge_gap=0.05, method="otsu",
                  window_sec=5.0, hop_sec=1.0):
    """
    Returns list of [start_time, end_time] call events in seconds.

    lo, hi      : bandpass edges (Hz), species-specific vocal range
    method      : "otsu" (default, robust to dense call bouts) or "mad"
                  (median + k*MAD noise floor; breaks down when calls occupy
                  most of the recording, since the "noise floor" then
                  includes call energy)
    k           : used only for method="mad": threshold = median + k*MAD
    min_dur     : drop events shorter than this (seconds) - removes clicks/blips
    merge_gap   : merge events separated by less than this (seconds) - bridges
                  brief within-call amplitude dips (e.g. multi-syllable notes)
    """
    yf = bandpass(y, sr, lo, hi)
    times, rms_db = energy_envelope_db(yf, sr)
    if method == "windowed_otsu":
        thresh = windowed_otsu_threshold(times, rms_db, window_sec=window_sec, hop_sec=hop_sec)
    elif method == "otsu":
        thresh = otsu_threshold(rms_db)
    else:
        med = np.median(rms_db)
        mad = np.median(np.abs(rms_db - med))
        thresh = med + k * mad
    above = rms_db > thresh  # works whether thresh is scalar or per-frame array

    events = []
    in_event = False
    start = None
    for i, a in enumerate(above):
        if a and not in_event:
            start = times[i]
            in_event = True
        elif not a and in_event:
            events.append([start, times[i]])
            in_event = False
    if in_event:
        events.append([start, times[-1]])

    merged = []
    for ev in events:
        if merged and ev[0] - merged[-1][1] < merge_gap:
            merged[-1][1] = ev[1]
        else:
            merged.append(list(ev))

    merged = [ev for ev in merged if ev[1] - ev[0] >= min_dur]
    return merged, times, rms_db, thresh


# Species-specific bandpass ranges, set from visual/spectral calibration.
# method="otsu" chosen over median+MAD after finding MAD breaks down on dense
# call bouts (calls occupying most of the recording pull the "noise floor"
# estimate up toward call level itself). See piha_225167 / crow_458184
# diagnostics.
# TODO: revisit once more exemplar files per species have been QC'd.
SPECIES_PARAMS = {
    "Carrion_Crow": dict(lo=800, hi=7000, method="otsu", min_dur=0.03, merge_gap=0.05),
    "Screaming_Piha": dict(lo=1500, hi=5500, method="otsu", min_dur=0.03, merge_gap=0.05),
    "Papio_ursinus": dict(lo=300, hi=4000, method="otsu", min_dur=0.03, merge_gap=0.05),
    "Chlorocebus_pygerythrus": dict(lo=500, hi=6000, method="otsu", min_dur=0.03, merge_gap=0.05),
    "Turdus_merula": dict(lo=1000, hi=7000, method="otsu", min_dur=0.03, merge_gap=0.05),
    "Serinus_canaria": dict(lo=2500, hi=7500, method="otsu", min_dur=0.03, merge_gap=0.05),
    "Zenaida_macroura": dict(lo=350, hi=900, method="otsu", min_dur=0.03, merge_gap=0.05),
    "Coturnix_japonica": dict(lo=300, hi=4000, method="otsu", min_dur=0.03, merge_gap=0.05),
    "Menura_novaehollandiae": dict(lo=1000, hi=7500, method="otsu", min_dur=0.03, merge_gap=0.05),
    "Pongo_pygmaeus": dict(lo=300, hi=4000, method="otsu", min_dur=0.03, merge_gap=0.05),
    "Cercopithecus_diana": dict(lo=300, hi=2000, method="otsu", min_dur=0.03, merge_gap=0.05),
    "Psittacus_erithacus": dict(lo=500, hi=7500, method="otsu", min_dur=0.03, merge_gap=0.05),
    "Lonchura_striata": dict(lo=4000, hi=7500, method="otsu", min_dur=0.03, merge_gap=0.05),
    "Macaca_cyclopis": dict(lo=2500, hi=5500, method="otsu", min_dur=0.03, merge_gap=0.05),
    "Macaca_fascicularis": dict(lo=800, hi=3500, method="otsu", min_dur=0.03, merge_gap=0.05),
    "Lemur_catta": dict(lo=800, hi=4000, method="otsu", min_dur=0.03, merge_gap=0.05),
}
