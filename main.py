import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

count = 0
hand_up = False

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            highest_hand_y = h

            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                wrist = hand_landmarks.landmark[
                    mp_hands.HandLandmark.WRIST
                ]

                y = int(wrist.y * h)

                if y < highest_hand_y:
                    highest_hand_y = y

            threshold = int(h * 0.4)

            if highest_hand_y < threshold and not hand_up:
                count += 1
                hand_up = True

            if highest_hand_y > threshold + 50:
                hand_up = False

            cv2.line(
                frame,
                (0, threshold),
                (w, threshold),
                (0, 255, 255),
                2
            )

        cv2.putText(
            frame,
            f"Liczba podniesien: {count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Licznik podniesien dloni", frame)

        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()