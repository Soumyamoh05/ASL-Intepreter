import os
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

DATA_DIR = "data/sequences"
SEQ_LEN = 20

X = []
y = []
label_map = {}

print("Loading sequence dataset...\n")

label_id = 0

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)

    label_map[label] = label_id
    label_id += 1

    for file in os.listdir(label_path):
        seq_path = os.path.join(label_path, file)

        seq = np.load(seq_path)

        if len(seq) == SEQ_LEN:
            X.append(seq)
            y.append(label_map[label])

print(f"Total sequences: {len(X)}")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = models.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 63)),
    layers.Dropout(0.3),

    layers.LSTM(64),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_map), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining LSTM model...\n")

history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_test, y_test),
    batch_size=16
)

loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ LSTM Accuracy: {acc*100:.2f}%")

os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.h5")
np.save("models/lstm_labels.npy", label_map)

print("\n✅ LSTM Model saved")