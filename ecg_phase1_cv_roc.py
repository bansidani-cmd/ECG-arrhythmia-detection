"""
BIOMECHATRONICS PROJECT - PART 1F: ECG Arrhythmia Detection
        PHASE 1: Cross-Validation, ROC/PR Curves, Model Saving

Upgrades the fixed 22/22 patient train/test split into full patient-
grouped k-fold cross-validation across all 44 patients, producing a
mean and standard deviation instead of one single result. Adds ROC
and precision-recall curves, and saves the final trained model for
reuse in the combined monitoring application.

REQUIREMENTS:
    pip install wfdb numpy scipy scikit-learn matplotlib joblib

"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)
import joblib

DATA_DIR = './mitdb'

ALL_RECORDS = ['101','106','108','109','112','114','115','116','118','119',
               '122','124','201','203','205','207','208','209','215','220','223','230',
               '100','103','105','111','113','117','121','123','200','202',
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

# 
# LOAD ALL 44 PATIENTS, TRACKING WHICH PATIENT EACH BEAT BELONGS TO
# 

print(f"Loading all {len(ALL_RECORDS)} patient records...")
X_list, y_list, patient_id_list = [], [], []
for rec in ALL_RECORDS:
    Xr, yr = load_record_features(rec)
    if Xr is None:
        continue
    X_list.append(Xr)
    y_list.append(yr)
    patient_id_list.extend([rec] * len(yr))
    print(f"  Record {rec}: {len(yr)} beats ({np.sum(yr==1)} abnormal)")

X = np.vstack(X_list)
y = np.concatenate(y_list)
patient_id = np.array(patient_id_list)

print(f"\nTotal beats: {len(y)} ({np.sum(y==1)} abnormal)")
print(f"Total patients: {len(set(patient_id))}")

# 
# PATIENT-GROUPED K-FOLD CROSS-VALIDATION
# 

N_FOLDS = 5
group_kfold = GroupKFold(n_splits=N_FOLDS)

fold_results = []
all_fold_probabilities = np.zeros(len(y))

print(f"\nRunning {N_FOLDS}-fold patient-grouped cross-validation...")
for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups=patient_id)):
    X_train_fold, y_train_fold = X[train_idx], y[train_idx]
    X_test_fold, y_test_fold = X[test_idx], y[test_idx]

    fold_clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    fold_clf.fit(X_train_fold, y_train_fold)

    fold_pred = fold_clf.predict(X_test_fold)
    fold_proba = fold_clf.predict_proba(X_test_fold)[:, 1]
    all_fold_probabilities[test_idx] = fold_proba

    report = classification_report(y_test_fold, fold_pred, target_names=['Normal', 'Abnormal'], output_dict=True)
    fold_results.append({
        'fold': fold_idx + 1,
        'abnormal_precision': report['Abnormal']['precision'],
        'abnormal_recall': report['Abnormal']['recall'],
        'abnormal_f1': report['Abnormal']['f1-score'],
    })
    print(f"  Fold {fold_idx+1}: precision={report['Abnormal']['precision']:.3f}, "
          f"recall={report['Abnormal']['recall']:.3f}, f1={report['Abnormal']['f1-score']:.3f}")

precisions = [r['abnormal_precision'] for r in fold_results]
recalls = [r['abnormal_recall'] for r in fold_results]
f1s = [r['abnormal_f1'] for r in fold_results]

print("\n=== Cross-Validation Summary (Abnormal class) ===")
print(f"Precision: {np.mean(precisions):.3f} +/- {np.std(precisions):.3f}")
print(f"Recall:    {np.mean(recalls):.3f} +/- {np.std(recalls):.3f}")
print(f"F1:        {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")

# 
# ROC AND PRECISION-RECALL CURVES
# 

fpr, tpr, _ = roc_curve(y, all_fold_probabilities)
roc_auc = roc_auc_score(y, all_fold_probabilities)

precision_curve, recall_curve, _ = precision_recall_curve(y, all_fold_probabilities)
avg_precision = average_precision_score(y, all_fold_probabilities)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random classifier')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve - ECG Arrhythmia Detection')
axes[0].legend()

axes[1].plot(recall_curve, precision_curve, color='mediumpurple', label=f'PR curve (AP = {avg_precision:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve - ECG Arrhythmia Detection')
axes[1].legend()

plt.tight_layout()
plt.savefig('ecg_roc_pr_curves.png', dpi=120)
plt.close()
print(f"\nROC AUC: {roc_auc:.3f}")
print(f"Average Precision: {avg_precision:.3f}")
print("Saved ecg_roc_pr_curves.png")

# 
# TRAIN FINAL MODEL ON ALL DATA AND SAVE
# 

final_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
final_model.fit(X, y)

joblib.dump(final_model, 'ecg_arrhythmia_model.joblib')
joblib.dump(['pre_rr', 'post_rr', 'amplitude', 'energy', 'qrs_width'], 'ecg_feature_columns.joblib')
print("\nSaved ecg_arrhythmia_model.joblib and ecg_feature_columns.joblib")

print("\nPhase 1 complete.")