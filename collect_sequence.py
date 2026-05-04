import cv2
import mediapipe as mp
import os
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    max_num_hands=1
)

SEQUENCE_LENGTH = 20
TOTAL_SEQUENCES = 50
DELAY_BETWEEN_SEQUENCES = 1.5  # seconds

while True:
    print("\n==== SEQUENCE COLLECTION ====")
    label = input("Enter motion word (HELLO / THANKYOU) or q to quit: ").upper()

    if label == 'Q':
        break

    save_path = f"data/sequences/{label}"
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)

    print(f"\nGet ready... Starting in 3 seconds for {label}")
    time.sleep(3)

    seq_count = 0

    while seq_count < TOTAL_SEQUENCES:
        sequence = []
        frame_count = 0

        print(f"\nRecording sequence {seq_count+1}/{TOTAL_SEQUENCES}")

        while frame_count < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                print("Camera error")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    data = []
                    for lm in hand_landmarks.landmark:
                        data.extend([lm.x, lm.y, lm.z])

                    sequence.append(data)
                    frame_count += 1

                    break  # only one hand

            # UI display
            cv2.putText(frame, f"{label}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(frame, f"Seq: {seq_count+1}/{TOTAL_SEQUENCES}", (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            cv2.putText(frame, f"Frame: {frame_count}/{SEQUENCE_LENGTH}", (20,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            cv2.imshow("Recording Sequence", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # Save only if full sequence collected
        if len(sequence) == SEQUENCE_LENGTH:
            file_name = os.path.join(save_path, f"seq_{seq_count}.npy")
            np.save(file_name, sequence)
            print(f"Saved: {file_name}")
            seq_count += 1

        # Small delay between sequences
        print("Next sequence starting...")
        time.sleep(DELAY_BETWEEN_SEQUENCES)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Done collecting {TOTAL_SEQUENCES} sequences for {label}")