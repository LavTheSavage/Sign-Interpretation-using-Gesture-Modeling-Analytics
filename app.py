import cv2
import pickle
import collections
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),           # Thumb
    (0,5), (5,6), (6,7), (7,8),           # Index
    (5,9), (9,10), (10,11), (11,12),      # Middle
    (9,13), (13,14), (14,15), (15,16),    # Ring
    (13,17), (17,18), (18,19), (19,20),   # Pinky
    (0,17)                                # Palm base
]

def extract_landmarks(hand_landmarks):
    base_x = hand_landmarks[0].x
    base_y = hand_landmarks[0].y
    base_z = hand_landmarks[0].z
    features = []
    for lm in hand_landmarks:
        features.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    return features


with open('sign_language_model.pkl', 'rb') as f:
    model = pickle.load(f)


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)


frame_sequence = collections.deque(maxlen=30)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    translation_text = "SWIPE OR MOVE HAND"
    confidence = 0.0

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        

        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        features = extract_landmarks(hand_landmarks)
        

        frame_sequence.append(features)

  
        if len(frame_sequence) == 30:
            flat_sequence = np.array(frame_sequence).flatten()
            prediction = model.predict([flat_sequence])
            probabilities = model.predict_proba([flat_sequence])

            translation_text = prediction[0]
            confidence = np.max(probabilities) * 100


        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, pts[start], pts[end], (255, 255, 0), 2, cv2.LINE_AA)


        for idx, pt in enumerate(pts):
            radius = 7 if idx in [4, 8, 12, 16, 20] else 4
            cv2.circle(frame, pt, radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, radius, (255, 0, 255), -1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
    
    if confidence > 65:
        display_str = f"TRANSLATION: {translation_text} ({confidence:.0f}%)"
        text_color = (0, 255, 0)
    else:
        display_str = "TRANSLATION: Listening..."
        text_color = (0, 165, 255)

    cv2.putText(frame, display_str, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

    cv2.imshow('Live Dynamic Sign Language Translator', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()