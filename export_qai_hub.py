"""
export_qai_hub.py
Exports the trained gesture classifier to ONNX and compiles/quantizes it for
a Snapdragon device via Qualcomm AI Hub.

Prerequisites:
    pip install qai-hub qai-hub-models
    qai-hub configure --api_token <YOUR_TOKEN>   # from aihub.qualcomm.com

Usage:
    python src/export_qai_hub.py --device "Snapdragon X Elite CRD"
"""

import argparse

import torch

import qai_hub as hub

from dataset import SEQUENCE_LENGTH, FEATURE_DIM
from model import SignGestureClassifier


def export_and_compile(checkpoint_path: str, device_name: str, onnx_path: str = "models/gesture_classifier.onnx"):
    # 1. Load trained PyTorch model
    model = SignGestureClassifier()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # 2. Export to ONNX
    dummy_input = torch.randn(1, SEQUENCE_LENGTH, FEATURE_DIM)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["landmark_sequence"],
        output_names=["gesture_logits"],
        opset_version=13,
    )
    print(f"Exported ONNX model to {onnx_path}")

    # 3. Submit to Qualcomm AI Hub for compilation targeting the chosen device
    device = hub.Device(device_name)
    compile_job = hub.submit_compile_job(
        model=onnx_path,
        device=device,
        options="--target_runtime qnn_context_binary --quantize_full_type int8",
    )
    compile_job.wait()
    print(f"Compile job status: {compile_job.get_status()}")

    # 4. Profile on a real Snapdragon device to confirm real-time latency
    profile_job = hub.submit_profile_job(
        model=compile_job.get_target_model(),
        device=device,
    )
    profile_job.wait()
    print("Profiling complete — check the AI Hub dashboard for latency/memory results.")

    # 5. Download the compiled, NPU-ready model
    target_model = compile_job.get_target_model()
    target_model.download("models/gesture_classifier_quantized.bin")
    print("Downloaded NPU-optimized model to models/gesture_classifier_quantized.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/gesture_classifier.pt")
    parser.add_argument("--device", default="Snapdragon X Elite CRD",
                         help="Target device name as listed on Qualcomm AI Hub")
    args = parser.parse_args()

    export_and_compile(args.checkpoint, args.device)
