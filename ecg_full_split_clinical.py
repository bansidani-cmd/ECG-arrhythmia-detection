"""
BIOMECHATRONICS PROJECT — PART 1D: ECG Arrhythmia Detection
              (Full Patient Split + Recall-Prioritized Threshold)
Two upgrades from the previous version:

1. FULL PATIENT SPLIT: using the standard ~22-patient-per-group
   inter-patient split from de Chazal et al. (2004), instead of
   smaller 6-patient subset. 

2. RECALL-PRIORITIZED THRESHOLD: instead of the default 50% confidence
   cutoff, lower the threshold for calling a beat "Abnormal."  This
   deliberately trades some precision for higher recall, mimicking how
   real clinical screening tools are tuned (better to flag too much
   than miss a real problem).

WARNING: this downloads 44 patient records total. This WILL take a
while — get a coffee. Progress prints as each patient loads.
=====================================================================
"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_DIR = './mitdb'

# Standard de Chazal et al. (2004) inter-patient split
# (excludes paced-beat records 102, 104, 107, 217 which need different handling)
TRAIN_RECORDS = ['101','106','108','109','112','114','115','116','118','119',
                  '122','124','201','203','205','207','208','209','215','220','223','230']
TEST_RECORDS  = ['100','103','105','111','113','117','121','123','200','202',
                  '210','212','213','214','219','221','222','228','231','232','233','234']

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
    try:
        wfdb.dl_database('mitdb', dl_dir=DATA_DIR, records=[record_name])
        record = wfdb.rdrecord(f'{DATA_DIR}/{record_name}')
        annotation = wfdb.rdann(f'{DATA_DIR}/{record_name}', 'atr')
    except Exception as e:
        print(f"  Skipping record {record_name} (error: {e})")
        return None, None

    fs = record.fs
    signal = record.p_signal[:, 0]
    filtered_signal = bandpass_filter(signal, fs=fs)

    beat_samples = annotation.sample
    beat_symbols = annotation.symbol
    labels = np.array([0 if s in NORMAL_SYMBOLS else 1 for s in beat_symbols])

    X = extract_features(filtered_signal, beat_samples, fs)
    valid_mask = ~np.isnan(X).any(axis=1)
    return X[valid_mask], labels[valid_mask]

def load_group(record_list, group_name):
    print(f"\nLoading {group_name} patients ({len(record_list)} records)...")
    X_list, y_list = [], []
    for rec in record_list:
        Xr, yr = load_record_features(rec)
        if Xr is None:
            continue
        X_list.append(Xr)
        y_list.append(yr)
        print(f"  Record {rec}: {len(yr)} beats ({np.sum(yr==1)} abnormal)")
    return np.vstack(X_list), np.concatenate(y_list)


# LOAD FULL TRAIN / TEST SETS
#

X_train, y_train = load_group(TRAIN_RECORDS, "TRAINING")
X_test, y_test = load_group(TEST_RECORDS, "TEST (unseen)")

print(f"\nTotal training beats: {len(y_train)} ({np.sum(y_train==1)} abnormal)")
print(f"Total test beats:     {len(y_test)} ({np.sum(y_test==1)} abnormal)")


# TRAIN
# 

clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)


# EVALUATE — DEFAULT THRESHOLD (0.5, the "neutral" cutoff)
# 

y_pred_default = clf.predict(X_test)

print("\n=== DEFAULT THRESHOLD (0.5) — Full Patient Split ===")
print(classification_report(y_test, y_pred_default, target_names=['Normal', 'Abnormal']))


# EVALUATE — RECALL-PRIORITIZED THRESHOLD
# 
# predict_proba gives  the model's confidence (0.0-1.0) that a beat
# is Abnormal, instead of a hard yes/no. Then choose cutoff
# instead of the default 0.5, deliberately biasing toward catching
# more true abnormal beats at the cost of more false alarms.

probabilities = clf.predict_proba(X_test)[:, 1]  # probability of "Abnormal"

THRESHOLD = 0.3  # lower = more sensitive, catches more, more false alarms
y_pred_sensitive = (probabilities >= THRESHOLD).astype(int)

print(f"\n=== RECALL-PRIORITIZED THRESHOLD ({THRESHOLD}) — Full Patient Split ===")
print(classification_report(y_test, y_pred_sensitive, target_names=['Normal', 'Abnormal']))

# Confusion matrix for the recall-prioritized version (the more clinically relevant one)
cm = confusion_matrix(y_test, y_pred_sensitive)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
disp.plot(cmap='Blues')
plt.title(f"Confusion Matrix — Full Split, Threshold={THRESHOLD}")
plt.tight_layout()
plt.savefig('full_split_sensitive_confusion_matrix.png', dpi=120)
plt.close()
print("\nSaved full_split_sensitive_confusion_matrix.png")

importances = clf.feature_importances_
feature_names = ['pre_rr', 'post_rr', 'amplitude', 'energy', 'qrs_width']
print("\n=== Feature Importances ===")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name:12s}: {imp:.3f}")

print("\nCompare the two classification reports above:")
print("  - Default threshold (0.5): balanced, but may miss abnormal beats")
print("  - Recall-prioritized (0.3): should show higher Abnormal recall,")
print("    at the cost of lower Abnormal precision (more false alarms)")
print("This tradeoff, chosen deliberately, is exactly how real clinical")
print("screening tools are tuned: better to over-flag than miss something.")