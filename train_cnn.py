import os
import cv2
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

DATA_DIR = "data/static"
IMG_SIZE = 64

X = []
y = []
label_map = {}

print("Loading static dataset...\n")

label_id = 0

for category in os.listdir(DATA_DIR):
    category_path = os.path.join(DATA_DIR, category)

    for label in os.listdir(category_path):
        label_path = os.path.join(category_path, label)

        if label not in label_map:
            label_map[label] = label_id
            label_id += 1

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label_map[label])

print(f"Total samples: {len(X)}")

X = np.array(X) / 255.0
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # 🔥 prevents overfitting
    layers.Dense(len(label_map), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining CNN model...\n")

history = model.fit(
    X_train, y_train,
    epochs=15,
    validation_data=(X_test, y_test),
    batch_size=32
)

loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ CNN Accuracy: {acc*100:.2f}%")

os.makedirs("models", exist_ok=True)
model.save("models/cnn_model.h5")
np.save("models/cnn_labels.npy", label_map)

print("\n✅ CNN Model saved")