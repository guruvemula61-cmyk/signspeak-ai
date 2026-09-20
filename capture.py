"""
capture.py
Webcam capture + MediaPipe hand/pose landmark extraction for SignSpeak AI.

Each frame is converted into a fixed-length landmark vector that downstream
the gesture classifier (model.py) consumes.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


class LandmarkExtractor:
    """Extracts hand + upper-body pose landmarks from a video frame."""

    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.6):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=0.5,
        )
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame_bgr: np.ndarray):
        """Returns a flattened landmark vector, or None if nothing detected."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        hand_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)

        if not hand_results.multi_hand_landmarks and not pose_results.pose_landmarks:
            return None

        vector = []

        hand_slots = [None, None]
        if hand_results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks[:2]):
                hand_slots[i] = hand_landmarks

        for hand_landmarks in hand_slots:
            if hand_landmarks is None:
                vector.extend([0.0] * 63)
            else:
                for lm in hand_landmarks.landmark:
                    vector.extend([lm.x, lm.y, lm.z])

        upper_body_indices = [11, 12, 13, 14, 15, 16]
        if pose_results.pose_landmarks:
            for idx in upper_body_indices:
                lm = pose_results.pose_landmarks.landmark[idx]
                vector.extend([lm.x, lm.y, lm.z])
        else:
            vector.extend([0.0] * (len(upper_body_indices) * 3))

        return np.array(vector, dtype=np.float32)

    def close(self):
        self.hands.close()
        self.pose.close()


def run_preview():
    """Quick manual test: shows webcam feed with landmark overlay."""
    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit.")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        vec = extractor.extract(frame)
        status = "Landmarks detected" if vec is not None else "No landmarks"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("SignSpeak AI - Landmark Preview", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    run_preview()
