import cv2
import csv
import os
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def extract_landmarks(hand_landmarks):
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y
    base_z = hand_landmarks[0].z
    
    features = []
    for lm in hand_landmarks:
        features.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    return features 


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)


label = input("Enter sign name (e.g., HELLO, WAVE, THANK_YOU): ").strip().upper()
NUM_SEQUENCES = 30  
SEQUENCE_LENGTH = 30  

csv_file = "gesture_data.csv"
cap = cv2.VideoCapture(0)

print(f"\n--- READY TO RECORD SIGN: '{label}' ---")
print("Press 'S' to start recording sequences. Press 'Q' to exit.")

recording = False
seq_count = 0

while cap.isOpened() and seq_count < NUM_SEQUENCES:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    cv2.putText(frame, f"SIGN: {label} | Recorded: {seq_count}/{NUM_SEQUENCES}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if not recording:
        cv2.putText(frame, "Press 'S' to Start Recording", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            recording = True
        elif key == ord('q'):
            break
        continue

  
    print(f"Recording Sequence {seq_count + 1}/{NUM_SEQUENCES} in 3... 2... 1...")
    time.sleep(0.5)

    sequence_data = []  
    frame_counter = 0

    while frame_counter < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            features = extract_landmarks(detection_result.hand_landmarks[0])
            sequence_data.extend(features)
            

            for lm in detection_result.hand_landmarks[0]:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

            frame_counter += 1

        cv2.putText(frame, f"RECORDING... Frame {frame_counter}/{SEQUENCE_LENGTH}", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Data Collector", frame)
        cv2.waitKey(1)

    if len(sequence_data) == SEQUENCE_LENGTH * 63:
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([label] + sequence_data)
        seq_count += 1

cap.release()
cv2.destroyAllWindows()
print(f"Finished! Successfully recorded {seq_count} sequences for '{label}'.")