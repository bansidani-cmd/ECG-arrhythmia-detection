"""
BIOMECHATRONICS PROJECT — PART 1C: ECG Arrhythmia Detection
                    (Inter-Patient + QRS Width Feature)

Same inter-patient evaluation as before, but now with a 5th feature:
QRS WIDTH. An estimate of how long the main heartbeat spike lasts.

WHY THIS FEATURE: abnormal beats originating in the ventricles (like
PVCs) spread electrically through the heart muscle the "slow way"
instead of through the heart's fast conduction pathways, which makes
their QRS complex measurably WIDER than a normal beat's. 

HOW ITS MEASURED: Take the "energy envelope" around each detected
peak (the same squared-derivative signal used for peak detection) and
measure its width at half its maximum height (FWHM). A simple,
standard way to estimate the duration of a pulse-like feature.

"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_DIR = './mitdb'
TRAIN_RECORDS = ['101', '106', '108', '109', '112', '114']
TEST_RECORDS  = ['100', '103', '105', '111', '113', '117']
NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}

def bandpass_filter(sig, lowcut=0.5, highcut=40.0, fs=360, order=2):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype='band')
    return filtfilt(b, a, sig)

def compute_qrs_width_ms(sig, peak, fs, window_sec=0.15):
    """
    Estimate QRS width (in milliseconds) using FWHM of the energy
    envelope around a beat.
    """
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
    width_ms = (width_samples / fs) * 1000
    return width_ms

def extract_features(sig, beat_samples, fs):
    features = []
    window = int(0.1 * fs)
    for i, peak in enumerate(beat_samples):
        pre_rr = peak - beat_samples[i-1] if i > 0 else np.nan
        post_rr = beat_samples[i+1] - peak if i < len(beat_samples)-1 else np.nan
        start, end = max(0, peak - window), min(len(sig), peak + window)
        beat_window = sig[start:end]
        amplitude = np.max(beat_window) - np.min(beat_window)
        energy = np.sum(beat_window ** 2)
        qrs_width = compute_qrs_width_ms(sig, peak, fs)
        features.append([pre_rr, post_rr, amplitude, energy, qrs_width])
    return np.array(features)

def load_record_features(record_name):
    wfdb.dl_database('mitdb', dl_dir=DATA_DIR, records=[record_name])
    record = wfdb.rdrecord(f'{DATA_DIR}/{record_name}')
    annotation = wfdb.rdann(f'{DATA_DIR}/{record_name}', 'atr')

    fs = record.fs
    signal = record.p_signal[:, 0]
    filtered_signal = bandpass_filter(signal, fs=fs)

    beat_samples = annotation.sample
    beat_symbols = annotation.symbol
    labels = np.array([0 if s in NORMAL_SYMBOLS else 1 for s in beat_symbols])

    X = extract_features(filtered_signal, beat_samples, fs)
    valid_mask = ~np.isnan(X).any(axis=1)
    return X[valid_mask], labels[valid_mask]

print("Loading TRAINING patients:", TRAIN_RECORDS)
X_train_list, y_train_list = [], []
for rec in TRAIN_RECORDS:
    Xr, yr = load_record_features(rec)
    X_train_list.append(Xr)
    y_train_list.append(yr)
    print(f"  Record {rec}: {len(yr)} beats ({np.sum(yr==1)} abnormal)")

X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)

print("\nLoading TEST patients (never seen during training):", TEST_RECORDS)
X_test_list, y_test_list = [], []
for rec in TEST_RECORDS:
    Xr, yr = load_record_features(rec)
    X_test_list.append(Xr)
    y_test_list.append(yr)
    print(f"  Record {rec}: {len(yr)} beats ({np.sum(yr==1)} abnormal)")

X_test = np.vstack(X_test_list)
y_test = np.concatenate(y_test_list)

print(f"\nTotal training beats: {len(y_train)} ({np.sum(y_train==1)} abnormal)")
print(f"Total test beats:     {len(y_test)} ({np.sum(y_test==1)} abnormal)")

clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n=== Inter-Patient Classification Report (WITH QRS width) ===")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix — Inter-Patient + QRS Width")
plt.tight_layout()
plt.savefig('interpatient_qrswidth_confusion_matrix.png', dpi=120)
plt.close()
print("Saved interpatient_qrswidth_confusion_matrix.png")

importances = clf.feature_importances_
feature_names = ['pre_rr', 'post_rr', 'amplitude', 'energy', 'qrs_width']
print("\n=== Feature Importances ===")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name:12s}: {imp:.3f}")

print("\nCompare this classification report to your previous run (without qrs_width).")
print("Did Abnormal precision/recall improve? Did qrs_width rank as an important feature?")