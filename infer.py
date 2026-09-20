"""
infer.py
End-to-end real-time pipeline: webcam -> landmarks -> gesture -> speech.

Usage:
    python src/infer.py
    python src/infer.py --checkpoint models/gesture_classifier.pt
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

from capture import LandmarkExtractor
from dataset import SEQUENCE_LENGTH, DATASET_LABELS
from model import SignGestureClassifier
from tts import OfflineTTS

CONFIDENCE_THRESHOLD = 0.75
COOLDOWN_SECONDS = 1.5  # avoid repeating the same spoken word every frame


def load_model(checkpoint_path: str) -> SignGestureClassifier:
    model = SignGestureClassifier()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def run(checkpoint_path: str):
    model = load_model(checkpoint_path)
    extractor = LandmarkExtractor()
    tts = OfflineTTS()

    buffer = deque(maxlen=SEQUENCE_LENGTH)
    last_spoken, last_spoken_time = None, 0.0

    cap = cv2.VideoCapture(0)
    print("SignSpeak AI running — press 'q' to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        landmarks = extractor.extract(frame)
        if landmarks is not None:
            buffer.append(landmarks)

        caption = ""
        if len(buffer) == SEQUENCE_LENGTH:
            sequence = np.stack(buffer)[None, :, :]  # (1, seq_len, feature_dim)
            with torch.no_grad():
                logits = model(torch.from_numpy(sequence.astype(np.float32)))
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = int(probs.argmax())
                confidence = float(probs[pred_idx])

            if confidence >= CONFIDENCE_THRESHOLD:
                word = DATASET_LABELS[pred_idx]
                caption = f"{word} ({confidence:.0%})"

                now = time.time()
                if word != last_spoken or (now - last_spoken_time) > COOLDOWN_SECONDS:
                    tts.speak(word.replace("_", " "))
                    last_spoken, last_spoken_time = word, now

        cv2.putText(frame, caption, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("SignSpeak AI", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/gesture_classifier.pt")
    args = parser.parse_args()

    run(args.checkpoint)
