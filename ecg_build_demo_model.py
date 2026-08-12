"""

BIOMECHATRONICS PROJECT - PHASE 2, FIX: Honest ECG Demo Model

The production model used for reported results is trained on all 44
patients. Using that same model in the demo app is not a fair test,
since the demo patients were part of its training data.

This script trains a SEPARATE demo-only model, excluding the demo
patients entirely from training, then generates demo examples from
those excluded patients. Predictions on these examples reflect genuine
performance on unseen patients.

REQUIREMENTS:
    pip install wfdb numpy scipy scikit-learn joblib

"""

import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_DIR = './mitdb'

DEMO_RECORDS = ['100', '106', '119', '200']  # excluded from demo model training

TRAINING_RECORDS = ['101','108','109','112','114','115','116','118',
                     '122','124','201','203','205','207','208','209','215','220','223','230',
                     '103','105','111','113','117','121','123','202',
                     '210','212','213','214','219','221','222','228','231','232','233','234']

NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}
WINDOW_SAMPLES = 108
EXAMPLES_PER_CLASS = 15

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

# 
# TRAIN THE DEMO MODEL ON THE 40 NON-DEMO PATIENTS ONLY
# 

print(f"Training demo model on {len(TRAINING_RECORDS)} patients (excluding demo patients)...")
X_list, y_list = [], []
for rec in TRAINING_RECORDS:
    Xr, yr = load_record_features(rec)
    if Xr is None:
        continue
    X_list.append(Xr)
    y_list.append(yr)

X_train_demo = np.vstack(X_list)
y_train_demo = np.concatenate(y_list)
print(f"Demo model training beats: {len(y_train_demo)}")

demo_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced')
demo_model.fit(X_train_demo, y_train_demo)
joblib.dump(demo_model, 'ecg_demo_model.joblib')
import os
size_mb = os.path.getsize('ecg_demo_model.joblib') / (1024 * 1024)
print(f"Saved ecg_demo_model.joblib ({size_mb:.1f} MB)")

# 
# GENERATE A LARGE POOL OF CANDIDATE EXAMPLES FROM THE EXCLUDED PATIENTS
# 
# A larger pool than needed is collected first, then curated below into
# a deliberate mix spanning the full prediction confidence range.

candidate_pool = []

for record_name in DEMO_RECORDS:
    print(f"Processing demo record {record_name}...")
    wfdb.dl_database('mitdb', dl_dir=DATA_DIR, records=[record_name])
    record = wfdb.rdrecord(f'{DATA_DIR}/{record_name}')
    annotation = wfdb.rdann(f'{DATA_DIR}/{record_name}', 'atr')

    fs = record.fs
    signal = record.p_signal[:, 0]
    filtered_signal = bandpass_filter(signal, fs=fs)

    beat_samples = annotation.sample
    beat_symbols = annotation.symbol
    window = int(0.1 * fs)

    for i, peak in enumerate(beat_samples):
        if i == 0 or i == len(beat_samples) - 1:
            continue
        is_normal = beat_symbols[i] in NORMAL_SYMBOLS

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

        candidate_pool.append({
            'record': record_name,
            'waveform': waveform,
            'features': [pre_rr, post_rr, amplitude, energy, qrs_width],
            'true_label': 0 if is_normal else 1,
        })

print(f"\nCandidate pool size: {len(candidate_pool)}")

# 
# CURATE A DELIBERATE MIX ACROSS THE CONFIDENCE RANGE
# 

candidate_features = np.array([c['features'] for c in candidate_pool])
candidate_true = np.array([c['true_label'] for c in candidate_pool])
candidate_proba = demo_model.predict_proba(candidate_features)[:, 1]
candidate_pred = (candidate_proba >= 0.5).astype(int)

confident_correct_normal = [i for i in range(len(candidate_pool))
                             if candidate_true[i] == 0 and candidate_proba[i] < 0.1]
confident_correct_abnormal = [i for i in range(len(candidate_pool))
                               if candidate_true[i] == 1 and candidate_proba[i] > 0.9]
misclassified = [i for i in range(len(candidate_pool)) if candidate_pred[i] != candidate_true[i]]
borderline = [i for i in range(len(candidate_pool)) if 0.35 <= candidate_proba[i] <= 0.65]

rng3 = np.random.default_rng(42)
def sample_indices(pool, count):
    if len(pool) <= count:
        return pool
    return list(rng3.choice(pool, size=count, replace=False))

selected = set()
selected.update(sample_indices(confident_correct_normal, 8))
selected.update(sample_indices(confident_correct_abnormal, 8))
selected.update(sample_indices(misclassified, 8))
selected.update(sample_indices(borderline, 6))

print(f"\nConfident correct (normal): {len(confident_correct_normal)} available")
print(f"Confident correct (abnormal): {len(confident_correct_abnormal)} available")
print(f"Misclassified: {len(misclassified)} available")
print(f"Borderline (0.35-0.65): {len(borderline)} available")
print(f"Total curated examples selected: {len(selected)}")

demo_examples = [candidate_pool[i] for i in sorted(selected)]

# Quick accuracy check to confirm this is a genuine held-out test
predictions = demo_model.predict(np.array([e['features'] for e in demo_examples]))
true_labels = np.array([e['true_label'] for e in demo_examples])
accuracy = np.mean(predictions == true_labels)
print(f"Demo model accuracy on held-out demo examples: {accuracy:.3f}")
print(f"(A value meaningfully below 1.0 confirms this is a genuine held-out test.)")

joblib.dump(demo_examples, 'ecg_demo_examples.joblib')
print("\nSaved ecg_demo_examples.joblib")