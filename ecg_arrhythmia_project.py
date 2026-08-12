"""
BIOMECHATRONICS PROJECT — PART 1: ECG Arrhythmia Detection
Dataset : MIT-BIH Arrhythmia Database (PhysioNet)
Goal    : Classify individual heartbeats as Normal vs Abnormal using signal processing + a simple machine learning model.


"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# STEP 1: DOWNLOAD ONE RECORD

# MIT-BIH has 48 records, numbered like 100, 101, 103, 105, 108, 200...
# Each record = ~30 minutes of ECG from one patient, sampled at 360 Hz,
# with every single heartbeat labeled by a cardiologist.

RECORD_NAME = '119'
DATA_DIR = './mitdb'   # files will be saved here

print("Downloading record", RECORD_NAME, "...")
wfdb.dl_database('mitdb', dl_dir=DATA_DIR, records=[RECORD_NAME])
print("Done.")


# STEP 2: LOAD THE RECORD + ANNOTATIONS

# wfdb.rdrecord() loads the raw signal.
# wfdb.rdann() loads the cardiologist's beat-by-beat annotations —
# this includes WHERE each heartbeat is (sample index) and WHAT TYPE
# it is (e.g. 'N' = normal, 'V' = premature ventricular contraction).

record = wfdb.rdrecord(f'{DATA_DIR}/{RECORD_NAME}')
annotation = wfdb.rdann(f'{DATA_DIR}/{RECORD_NAME}', 'atr')

fs = record.fs  # sampling frequency (Hz) — 360 for MIT-BIH
signal = record.p_signal[:, 0]  # use the first ECG channel

print(f"Signal length: {len(signal)} samples ({len(signal)/fs:.1f} seconds)")
print(f"Sampling rate: {fs} Hz")
print(f"Number of annotated beats: {len(annotation.sample)}")

# Quick look at the raw signal (first 10 seconds)
plt.figure(figsize=(12, 3))
plt.plot(signal[:fs*10])
plt.title("Raw ECG signal (first 10 seconds)")
plt.xlabel("Sample")
plt.ylabel("Amplitude (mV)")
plt.tight_layout()
plt.savefig('step2_raw_signal.png', dpi=120)
plt.close()
print("Saved step2_raw_signal.png — look at this to see what raw ECG looks like.")


# STEP 3: FILTER THE SIGNAL

# Raw ECG has two common problems:
#   - Baseline wander: slow drift from breathing/movement (low freq)
#   - High-frequency noise: muscle activity, electrical interference
#
# A bandpass filter (here: 0.5–40 Hz) removes both, keeping the
# frequency range where actual heartbeat information lives.

def bandpass_filter(sig, lowcut=0.5, highcut=40.0, fs=360, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, sig)

filtered_signal = bandpass_filter(signal, fs=fs)

plt.figure(figsize=(12, 3))
plt.plot(signal[:fs*10], alpha=0.5, label='Raw')
plt.plot(filtered_signal[:fs*10], label='Filtered')
plt.legend()
plt.title("Raw vs Filtered ECG (first 10 seconds)")
plt.tight_layout()
plt.savefig('step3_filtered_signal.png', dpi=120)
plt.close()
print("Saved step3_filtered_signal.png — compare raw vs filtered.")


# STEP 4: R-PEAK DETECTION (simplified Pan-Tompkins style)

# The R-peak is the tall spike in each heartbeat — finding it tells
# WHEN each beat happens, which is the foundation for everything else
# (heart rate, RR intervals, beat segmentation).
#
# Real Pan-Tompkins involves: derivative -> squaring -> moving window
# integration -> adaptive thresholding. 

from scipy.signal import find_peaks

def detect_r_peaks(sig, fs):
    # 1. Derivative emphasizes the steep slope of the QRS complex
    diff_sig = np.diff(sig)
    # 2. Squaring makes all values positive and emphasizes large slopes
    squared = diff_sig ** 2
    # 3. Moving average "integrates" energy over a short window
    window_size = int(0.08 * fs)  # ~80ms window
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    # 4. Peak detection with a minimum distance between beats
    #    (a healthy heart rarely beats faster than ~220 bpm -> min ~270ms apart)
    min_distance = int(0.27 * fs)
    peaks, _ = find_peaks(integrated, distance=min_distance,
                           height=np.mean(integrated) * 1.2)
    return peaks

detected_peaks = detect_r_peaks(filtered_signal, fs)
print(f"Detected {len(detected_peaks)} R-peaks (annotation file has {len(annotation.sample)})")

# Visualize detected peaks over a short segment
plt.figure(figsize=(12, 3))
segment_end = fs * 10
plt.plot(filtered_signal[:segment_end])
peaks_in_view = detected_peaks[detected_peaks < segment_end]
plt.plot(peaks_in_view, filtered_signal[peaks_in_view], 'rx', markersize=10, label='Detected R-peak')
plt.legend()
plt.title("Detected R-peaks (first 10 seconds)")
plt.tight_layout()
plt.savefig('step4_r_peaks.png', dpi=120)
plt.close()
print("Saved step4_r_peaks.png — check the red X's line up with the spikes.")


# STEP 5: USE GROUND-TRUTH ANNOTATIONS FOR LABELS

# For LEARNING R-peak detection, own detector is great.
# For TRAINING A CLASSIFIER, want reliable ground truth — so from
# here on use the cardiologist's annotated peak locations and beat
# labels directly. This avoids mixing detector errors into  model.
#
# AAMI standard groups MIT-BIH's ~15 beat symbols into 5 classes.
# Simplify further into a binary problem: Normal vs Abnormal.

NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}  # normal + normal variants
# Everything else (V=PVC, A=APC, F=fusion, etc.) counts as Abnormal

beat_samples = annotation.sample
beat_symbols = annotation.symbol

labels = np.array([0 if s in NORMAL_SYMBOLS else 1 for s in beat_symbols])
print(f"Normal beats: {np.sum(labels==0)}, Abnormal beats: {np.sum(labels==1)}")


# STEP 6: FEATURE EXTRACTION PER BEAT
#
# For each annotated beat, compute a handful of numeric features
# that describe it. This turns "a chunk of raw signal" into something
# a machine learning model can actually learn from.
#
# Features used:
#   - pre_rr:  time (in samples) since the previous beat
#   - post_rr: time (in samples) until the next beat
#   - beat_amplitude: max value in a small window around the peak
#   - beat_energy: sum of squared signal in that window (~"how big" the beat is)

def extract_features(sig, beat_samples, fs):
    features = []
    window = int(0.1 * fs)  # 100ms window around each peak
    for i, peak in enumerate(beat_samples):
        pre_rr = peak - beat_samples[i-1] if i > 0 else np.nan
        post_rr = beat_samples[i+1] - peak if i < len(beat_samples)-1 else np.nan

        start = max(0, peak - window)
        end = min(len(sig), peak + window)
        beat_window = sig[start:end]

        amplitude = np.max(beat_window) - np.min(beat_window)
        energy = np.sum(beat_window ** 2)

        features.append([pre_rr, post_rr, amplitude, energy])
    return np.array(features)

X = extract_features(filtered_signal, beat_samples, fs)

# Drop the first and last beat (no valid pre_rr / post_rr) and matching labels
valid_mask = ~np.isnan(X).any(axis=1)
X = X[valid_mask]
y = labels[valid_mask]

print(f"Feature matrix shape: {X.shape}  (rows=beats, cols=features)")


# STEP 7: TRAIN / TEST SPLIT + CLASSIFIER
#
# stratify=y keeps the Normal/Abnormal ratio balanced in both splits —
# important since Normal beats vastly outnumber Abnormal ones.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)


# STEP 8: EVALUATE

# Accuracy alone can be misleading here (most beats are Normal).
# Care much more about RECALL on the Abnormal class — i.e.
# how many true abnormal beats were caught? 

y_pred = clf.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix — ECG Beat Classification")
plt.tight_layout()
plt.savefig('step8_confusion_matrix.png', dpi=120)
plt.close()
print("Saved step8_confusion_matrix.png")

# Feature importance — which features mattered most?
importances = clf.feature_importances_
feature_names = ['pre_rr', 'post_rr', 'amplitude', 'energy']
print("\n=== Feature Importances ===")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name:12s}: {imp:.3f}")

print("\nDone! You now have a full ECG beat-classification pipeline.")
print("Next steps to try:")
print("  - Run this on more records (e.g. '101', '106', '119', '200')")
print("    and combine them for a bigger, more general dataset.")
print("  - Add more features (e.g. QRS width, T-wave amplitude).")
print("  - Try multi-class classification (Normal / PVC / APC / Fusion)")
print("    instead of binary Normal/Abnormal.")