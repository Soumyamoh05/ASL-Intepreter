import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque, Counter

cnn_model = load_model("models/cnn_model.h5")
lstm_model = load_model("models/lstm_model.h5")

cnn_labels = np.load("models/cnn_labels.npy", allow_pickle=True).item()
cnn_labels = {v: k for k, v in cnn_labels.items()}

lstm_labels = np.load("models/lstm_labels.npy", allow_pickle=True).item()
lstm_labels = {v: k for k, v in lstm_labels.items()}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

sequence = deque(maxlen=20)   # for LSTM
pred_buffer = deque(maxlen=10)  # smoothing

final_prediction = ""

def detect_motion(seq):
    if len(seq) < 2:
        return False
    return np.linalg.norm(np.array(seq[-1]) - np.array(seq[0])) > 0.05


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_pred = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            sequence.append(data)

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            xmin = int(min(x_coords) * w) - 20
            xmax = int(max(x_coords) * w) + 20
            ymin = int(min(y_coords) * h) - 20
            ymax = int(max(y_coords) * h) + 20

            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)

            hand_img = frame[ymin:ymax, xmin:xmax]

            if hand_img.size != 0:
                img = cv2.resize(hand_img, (64, 64))
                img = np.expand_dims(img, axis=0) / 255.0

                cnn_pred = cnn_model.predict(img, verbose=0)
                cnn_label = cnn_labels[np.argmax(cnn_pred)]
                cnn_conf = np.max(cnn_pred)

                if len(sequence) == 20:
                    seq_array = np.array(sequence)
                    lstm_pred = lstm_model.predict(
                        np.expand_dims(seq_array, axis=0),
                        verbose=0
                    )
                    lstm_label = lstm_labels[np.argmax(lstm_pred)]
                    lstm_conf = np.max(lstm_pred)

                    if detect_motion(sequence) and lstm_conf > 0.8:
                        current_pred = lstm_label
                    else:
                        if cnn_conf > 0.7:
                            current_pred = cnn_label

                else:
                    if cnn_conf > 0.7:
                        current_pred = cnn_label

            break

    if current_pred != "":
        pred_buffer.append(current_pred)

    if len(pred_buffer) == 10:
        final_prediction = Counter(pred_buffer).most_common(1)[0][0]

    output_box = np.ones((100, w, 3), dtype=np.uint8) * 255

    cv2.putText(
        output_box,
        f"Prediction: {final_prediction}",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        3
    )

    combined = np.vstack((frame, output_box))

    cv2.imshow("ASL Hybrid Interpreter", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()