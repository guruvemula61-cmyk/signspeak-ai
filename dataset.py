"""
dataset.py
Loads and preprocesses ISL gesture data for training the classifier.

Expected layout (place raw videos/landmarks under data/raw/<label>/*.mp4
or pre-extracted landmark .npy sequences under data/processed/<label>/*.npy):

    data/
      raw/
        hello/
        thank_you/
        water/
        ...
      processed/
        hello/0001.npy
        ...

Swap DATASET_LABELS and the loading logic below for whichever open ISL
dataset you use (e.g. INCLUDE, ISL-CSLTR) or your own recorded vocabulary.
"""

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

SEQUENCE_LENGTH = 40  # frames per gesture clip, pad/truncate to this length
FEATURE_DIM = 144     # 2 * 21 * 3 (hands) + 6 * 3 (pose) -- matches capture.py


DATASET_LABELS = [
    "hello", "thank_you", "please", "water", "food", "help",
    "yes", "no", "name", "sorry",
    # extend with your target vocabulary
]


def pad_or_truncate(sequence: np.ndarray, length: int = SEQUENCE_LENGTH) -> np.ndarray:
    if len(sequence) >= length:
        return sequence[:length]
    pad = np.zeros((length - len(sequence), sequence.shape[1]), dtype=np.float32)
    return np.vstack([sequence, pad])


class ISLGestureDataset(Dataset):
    """Loads pre-extracted landmark sequences (.npy) organized by label folder."""

    def __init__(self, processed_dir: str = "data/processed", labels=None):
        self.processed_dir = Path(processed_dir)
        self.labels = labels or DATASET_LABELS
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.samples = self._index_samples()

    def _index_samples(self):
        samples = []
        for label in self.labels:
            label_dir = self.processed_dir / label
            if not label_dir.exists():
                continue
            for npy_file in label_dir.glob("*.npy"):
                samples.append((npy_file, self.label_to_idx[label]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx = self.samples[idx]
        sequence = np.load(path).astype(np.float32)
        sequence = pad_or_truncate(sequence)
        return sequence, label_idx


if __name__ == "__main__":
    ds = ISLGestureDataset()
    print(f"Found {len(ds)} samples across {len(DATASET_LABELS)} labels.")
    if len(ds) == 0:
        print(
            "No data found. Populate data/processed/<label>/*.npy using "
            "capture.py + a landmark-recording script, or point this at an "
            "open ISL dataset such as INCLUDE or ISL-CSLTR."
        )
