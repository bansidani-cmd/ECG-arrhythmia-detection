"""

BIOMECHATRONICS PROJECT — PART 1B: ECG Arrhythmia Detection
    (Inter-Patient Evaluation)

This is the upgraded, research-grade version of the ECG project.


"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_DIR = './mitdb'


# STEP 1: DEFINE TRAIN vs TEST PATIENT GROUPS
# 
# These records are split so that NO patient appears in both groups.
# (A smaller subset of the standard DS1/DS2 split from published
# research, chosen to keep download/runtime reasonable for a first run.)

TRAIN_RECORDS = ['101', '106', '108', '109', '112', '114']
TEST_RECORDS  = ['100', '103', '105', '111', '113', '117']

NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}


# STEP 2: REUSABLE FUNCTIONS (same logic as the first script)
# 

def bandpass_filter(sig, lowcut=0.5, highcut=40.0, fs=360, order=2):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype='band')
    return filtfilt(b, a, sig)

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
        features.append([pre_rr, post_rr, amplitude, energy])
    return np.array(features)

def load_record_features(record_name):
    """Download (if needed), load, filter, and extract features+labels for one record."""
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


# STEP 3: BUILD TRAIN AND TEST SETS FROM SEPARATE PATIENT GROUPS
# 

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


# STEP 4: TRAIN
# 

clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)


# STEP 5: EVALUATE ON COMPLETELY UNSEEN PATIENTS
# 

y_pred = clf.predict(X_test)

print("\n=== Inter-Patient Classification Report ===")
print("(Model has NEVER seen these patients before)")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix — Inter-Patient ECG Classification")
plt.tight_layout()
plt.savefig('interpatient_confusion_matrix.png', dpi=120)
plt.close()
print("Saved interpatient_confusion_matrix.png")

importances = clf.feature_importances_
feature_names = ['pre_rr', 'post_rr', 'amplitude', 'energy']
print("\n=== Feature Importances ===")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name:12s}: {imp:.3f}")

print("\nDone. This result is the honest, realistic performance number —")
print("compare it to your first script's 100% and you'll see the difference that inter-patient evaluation makes.")