"""
BIOMECHATRONICS PROJECT — PART 1E: ECG Arrhythmia Detection
                    (1D CNN on Raw Beat Waveforms)

Same inter-patient methodology (train/test on disjoint patients), but
now the model learns directly from the RAW WAVEFORM around each beat
instead of 5 hand-crafted numbers. This lets it discover its own
shape-based patterns instead of relying on features we guessed at.

"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = './mitdb'
TRAIN_RECORDS = ['101','106','108','109','112','114','115','116','118','119',
                  '122','124','201','203','205','207','208','209','215','220','223','230']
TEST_RECORDS  = ['100','103','105','111','113','117','121','123','200','202',
                  '210','212','213','214','219','221','222','228','231','232','233','234']

NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}
WINDOW_SAMPLES = 108  # samples before AND after the R-peak (so 216 total per beat)

def bandpass_filter(sig, lowcut=0.5, highcut=40.0, fs=360, order=2):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype='band')
    return filtfilt(b, a, sig)

def extract_raw_windows(sig, beat_samples, window=WINDOW_SAMPLES):
    """
    Instead of calculating summary numbers, this just slices out the
    raw voltage values around each R-peak. Each beat becomes a small
    array of shape (216,) instead of a handful of features.
    """
    windows = []
    valid_indices = []
    for i, peak in enumerate(beat_samples):
        start, end = peak - window, peak + window
        if start < 0 or end > len(sig):
            continue  # skip beats too close to the recording's edge
        beat_window = sig[start:end]
        # Normalize each beat individually (zero mean, unit variance) —
        # this matters a lot for neural nets, since it keeps every beat
        # on a comparable scale regardless of that patient's baseline signal strength.
        beat_window = (beat_window - np.mean(beat_window)) / (np.std(beat_window) + 1e-8)
        windows.append(beat_window)
        valid_indices.append(i)
    return np.array(windows), valid_indices

def load_record_raw(record_name):
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
    all_labels = np.array([0 if s in NORMAL_SYMBOLS else 1 for s in beat_symbols])

    X, valid_indices = extract_raw_windows(filtered_signal, beat_samples)
    y = all_labels[valid_indices]
    return X, y

def load_group_raw(record_list, group_name):
    print(f"\nLoading {group_name} patients ({len(record_list)} records)...")
    X_list, y_list = [], []
    for rec in record_list:
        Xr, yr = load_record_raw(rec)
        if Xr is None:
            continue
        X_list.append(Xr)
        y_list.append(yr)
        print(f"  Record {rec}: {len(yr)} beats ({np.sum(yr==1)} abnormal)")
    return np.vstack(X_list), np.concatenate(y_list)

# =====================================================================
# LOAD DATA
# =====================================================================

X_train_full, y_train_full = load_group_raw(TRAIN_RECORDS, "TRAINING")
X_test, y_test = load_group_raw(TEST_RECORDS, "TEST (unseen)")

print(f"\nTotal training beats: {len(y_train_full)} ({np.sum(y_train_full==1)} abnormal)")
print(f"Total test beats:     {len(y_test)} ({np.sum(y_test==1)} abnormal)")

# Hold out a small validation slice from TRAINING patients only (never touches
# test patients) so we can monitor the model during training without cheating.
val_split = int(0.9 * len(X_train_full))
rng = np.random.default_rng(42)
shuffle_idx = rng.permutation(len(X_train_full))
X_train_full, y_train_full = X_train_full[shuffle_idx], y_train_full[shuffle_idx]
X_train, y_train = X_train_full[:val_split], y_train_full[:val_split]
X_val, y_val = X_train_full[val_split:], y_train_full[val_split:]

# CNNs expect shape (num_samples, timesteps, channels) — we have 1 channel (single-lead ECG)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

# 
# HANDLE CLASS IMBALANCE
# 


class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\nClass weights (to counter imbalance): {class_weight_dict}")

# 
# BUILD THE 1D CNN
# python ecg_cnn_project.py
# Layer-by-layer intuition:
#   Conv1D    -> slides small filters across the waveform, learning to
#                detect useful local shape patterns
#   MaxPooling-> shrinks the signal, keeping only the strongest detected
#                pattern activations (also reduces overfitting risk)
#   (repeat, so later layers combine earlier patterns into more complex ones)
#   GlobalAveragePooling1D -> collapses the whole sequence into one summary
#                vector per filter, regardless of beat length
#   Dense     -> combines everything into a final decision
#   Dropout   -> randomly "turns off" neurons during training, forcing the
#                network to not over-rely on any single pattern (reduces overfitting)
#   sigmoid output -> squashes the final number into a 0-1 probability

model = keras.Sequential([
    layers.Input(shape=(WINDOW_SAMPLES * 2, 1)),
    layers.Conv1D(16, kernel_size=7, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(32, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# =====================================================================
# TRAIN
# =====================================================================
# epochs = how many full passes through the training data
# batch_size = how many beats the model looks at before updating itself

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=64,
    class_weight=class_weight_dict,
    verbose=1
)

# Plot training curves — a good sanity check for overfitting
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Training Loss')
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Training Accuracy')
plt.tight_layout()
plt.savefig('cnn_training_curves.png', dpi=120)
plt.close()
print("Saved cnn_training_curves.png")

# =====================================================================
# EVALUATE ON UNSEEN TEST PATIENTS
# =====================================================================

test_probabilities = model.predict(X_test_cnn).flatten()

print("\n=== CNN — DEFAULT THRESHOLD (0.5) — Full Patient Split ===")
y_pred_default = (test_probabilities >= 0.5).astype(int)
print(classification_report(y_test, y_pred_default, target_names=['Normal', 'Abnormal']))

THRESHOLD = 0.3
print(f"\n=== CNN — RECALL-PRIORITIZED THRESHOLD ({THRESHOLD}) — Full Patient Split ===")
y_pred_sensitive = (test_probabilities >= THRESHOLD).astype(int)
print(classification_report(y_test, y_pred_sensitive, target_names=['Normal', 'Abnormal']))

cm = confusion_matrix(y_test, y_pred_sensitive)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
disp.plot(cmap='Greens')
plt.title(f"CNN Confusion Matrix — Full Split, Threshold={THRESHOLD}")
plt.tight_layout()
plt.savefig('cnn_confusion_matrix.png', dpi=120)
plt.close()
print("\nSaved cnn_confusion_matrix.png")

print("\nDone. Compare this to your Random Forest full-split results:")
print("  Random Forest (0.3 threshold): precision 0.47, recall 0.82, F1 0.59")
print("Did the CNN's raw-waveform approach beat the hand-crafted features?")