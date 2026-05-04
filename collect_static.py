import cv2
import mediapipe as mp
import os
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    max_num_hands=1
)

IMG_SIZE = 64
MAX_IMAGES = 200

while True:
    print("\n==== DATA COLLECTION ====")
    choice = input("Enter 1 for Alphabets, 2 for Words, q to quit: ")

    if choice == 'q':
        print("Exiting...")
        break

    if choice not in ['1', '2']:
        print("Invalid input")
        continue

    category = "alphabets" if choice == '1' else "words"
    label = input("Enter label: ").upper()

    # 📁 Create correct folder
    save_path = f"data/static/{category}/{label}"
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)

    count = 0
    print(f"\nCollecting 200 images for {label}...")
    print("Get ready...")
    time.sleep(2)

    while count < MAX_IMAGES:
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

                # Draw landmarks (visual feedback)
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # 🔹 Get bounding box
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                xmin = int(min(x_coords) * w) - 20
                xmax = int(max(x_coords) * w) + 20
                ymin = int(min(y_coords) * h) - 20
                ymax = int(max(y_coords) * h) + 20

                # 🔹 Clamp values
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmax)
                ymax = min(h, ymax)

                hand_img = frame[ymin:ymax, xmin:xmax]

                # 🔹 Ensure valid crop
                if hand_img.size != 0:
                    try:
                        hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))

                        img_name = os.path.join(save_path, f"{count}.jpg")
                        cv2.imwrite(img_name, hand_img)

                        count += 1
                        print(f"{label}: {count}/200")

                        time.sleep(0.08)  # small delay

                    except:
                        pass

                break  # only one hand

        # Display info
        cv2.putText(frame, f"{label} ({count}/200)", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Collecting Static Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Done collecting for {label}")