"""

BIOMECHATRONICS PROJECT - PHASE 2, PREP: Export ECG Demo Examples

Selects a small set of example heartbeats (raw waveform, computed
features, and true label) from a handful of records and saves them
to disk. The combined monitoring app loads this file directly rather
than reprocessing the full dataset on every launch.

REQUIREMENTS:
    pip install wfdb numpy scipy joblib

"""

import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
import joblib

DATA_DIR = './mitdb'
DEMO_RECORDS = ['100', '106', '119', '200']  # a mix of easier and harder patients
EXAMPLES_PER_CLASS = 15
WINDOW_SAMPLES = 108  # matches the CNN raw window size, for waveform plotting

NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}

def bandpass_filter(sig, lowcut=0.5, highcut=40.0, fs=360, order=2):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype='band')
    return filtfilt(b, a, sig)

def compute_qrs_width_ms(sig, peak, fs, window_sec=0.15):
    diff_sig = np.diff(sig)
    squared = diff_sig ** 2
    win_samples = int(window_sec * fs)
    start = max(0, peak - win_samples)
    end = min(len(squared), peak + win_samples)
    segment = squared[start:end]
    if len(segment) < 3:
        return np.nan
    peak_val = np.max(segment)
    if peak_val == 0:
        return np.nan
    half_max = peak_val / 2
    above_half = np.where(segment > half_max)[0]
    if len(above_half) == 0:
        return np.nan
    width_samples = above_half[-1] - above_half[0]
    return (width_samples / fs) * 1000

demo_examples = []

for record_name in DEMO_RECORDS:
    print(f"Processing record {record_name}...")
    wfdb.dl_database('mitdb', dl_dir=DATA_DIR, records=[record_name])
    record = wfdb.rdrecord(f'{DATA_DIR}/{record_name}')
    annotation = wfdb.rdann(f'{DATA_DIR}/{record_name}', 'atr')

    fs = record.fs
    signal = record.p_signal[:, 0]
    filtered_signal = bandpass_filter(signal, fs=fs)

    beat_samples = annotation.sample
    beat_symbols = annotation.symbol
    window = int(0.1 * fs)

    normal_count, abnormal_count = 0, 0
    for i, peak in enumerate(beat_samples):
        if i == 0 or i == len(beat_samples) - 1:
            continue
        is_normal = beat_symbols[i] in NORMAL_SYMBOLS
        if is_normal and normal_count >= EXAMPLES_PER_CLASS:
            continue
        if not is_normal and abnormal_count >= EXAMPLES_PER_CLASS:
            continue

        start, end = peak - WINDOW_SAMPLES, peak + WINDOW_SAMPLES
        if start < 0 or end > len(filtered_signal):
            continue

        pre_rr = peak - beat_samples[i-1]
        post_rr = beat_samples[i+1] - peak
        beat_window_narrow = filtered_signal[max(0, peak-window):min(len(filtered_signal), peak+window)]
        amplitude = np.max(beat_window_narrow) - np.min(beat_window_narrow)
        energy = np.sum(beat_window_narrow ** 2)
        qrs_width = compute_qrs_width_ms(filtered_signal, peak, fs)

        if np.isnan(qrs_width):
            continue

        waveform = filtered_signal[start:end]

        demo_examples.append({
            'record': record_name,
            'waveform': waveform,
            'features': [pre_rr, post_rr, amplitude, energy, qrs_width],
            'true_label': 0 if is_normal else 1,
        })

        if is_normal:
            normal_count += 1
        else:
            abnormal_count += 1

print(f"\nTotal demo examples collected: {len(demo_examples)}")
print(f"Normal: {sum(1 for e in demo_examples if e['true_label']==0)}")
print(f"Abnormal: {sum(1 for e in demo_examples if e['true_label']==1)}")

joblib.dump(demo_examples, 'ecg_demo_examples.joblib')
print("\nSaved ecg_demo_examples.joblib")