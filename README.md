# SignSpeak AI
### Real-Time Indian Sign Language to Speech Translator on Snapdragon NPU

Submission for the **Snapdragon® AI Lab Build & Present Challenge** (Qualcomm).

SignSpeak AI captures Indian Sign Language (ISL) gestures through a webcam and
translates them into spoken and written language in real time — entirely
on-device, using the Snapdragon Hexagon NPU via Qualcomm AI Hub. No video or
audio ever leaves the device.

---

## Problem

India has an estimated 18 million deaf or hard-of-hearing citizens but fewer
than 300 certified ISL interpreters. Existing translation tools target
American Sign Language and rely on cloud APIs, which raises privacy concerns
and fails without internet access.

## Solution

A three-stage on-device pipeline:

1. **Landmark extraction** — hand/pose keypoints from each webcam frame
   (MediaPipe Hands / Pose).
2. **Gesture classification** — a lightweight quantized CNN-LSTM model maps
   the landmark sequence to a word/phrase, trained on open ISL datasets
   (INCLUDE, ISL-CSLTR).
3. **Speech output** — recognized text is captioned on-screen and converted
   to speech with an on-device TTS model.

All three stages are exported and quantized (INT8/FP16) through
**Qualcomm AI Hub** and profiled on a Snapdragon X-series device for
real-time latency.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Train the gesture classifier (after placing dataset in data/)
python src/train.py

# 2. Export & quantize for Snapdragon NPU via Qualcomm AI Hub
python src/export_qai_hub.py --device "Snapdragon X Elite CRD"

# 3. Run real-time webcam-to-speech translation
python src/infer.py
```

## Hardware Target

Optimized and profiled for Snapdragon X Elite / X2-powered HP PCs using the
Hexagon NPU (INT8/FP16 inference) through Qualcomm AI Hub.

## License

MIT — see `LICENSE`.
