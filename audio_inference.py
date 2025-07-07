import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_labeling_tools import detect_onsets

# ----- constants -----
MAX_T = 16  # same as in the training notebook
LABELS = ["hit", "squeak", "commentator", "other"]


# ----- feature helpers -----
def mfcc_stack(y, sr, n_mfcc=25):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=int(sr * 0.02), hop_length=int(sr * 0.02))
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, d1, d2])


def pad(feat):
    feat = feat[:, :MAX_T]
    if feat.shape[1] < MAX_T:
        pad_width = np.zeros((75, MAX_T - feat.shape[1]))
        feat = np.hstack([feat, pad_width])
    return feat[None]  # shape (1, 75, MAX_T)


# ----- network definition -----
class FourClassNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 24, (5, 5), padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, (5, 5), padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 48, (5, 5), padding=2), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * (75 // 4) * (MAX_T // 4), 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ----- utilities -----
def load_model(path: Path) -> FourClassNet:
    model = FourClassNet()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def analyze_audio(model: FourClassNet, wav_path: Path, sr: int = 48_000,
                  min_gap_sec: float = 0.25) -> list[dict]:
    y, sr_native = librosa.load(wav_path, sr=None, mono=True)
    if sr != sr_native:
        y = librosa.resample(y, sr_native, sr)

    onsets = detect_onsets(y, sr, hop_len=int(sr * 0.025),
                           delta=0.2, min_gap_sec=min_gap_sec)
    results = []
    pre = int(0.15 * sr)
    post = int(0.15 * sr)
    for s in onsets:
        start = max(s - pre, 0)
        end = min(s + post, len(y))
        clip = y[start:end]
        feat = torch.tensor(pad(mfcc_stack(clip, sr)), dtype=torch.float32)
        with torch.no_grad():
            probs = F.softmax(model(feat.unsqueeze(0)), dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        results.append({
            "time_sec": round(s / sr, 3),
            "label": LABELS[idx],
            "prob": float(probs[idx]),
        })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect events using the acoustic classifier")
    parser.add_argument("--audio", required=True, help="Path to WAV file")
    parser.add_argument("--model", default="models/acoustic_classifier.pth",
                        help="Path to model weights")
    parser.add_argument("--output", default="detections.json",
                        help="Where to write JSON results")
    args = parser.parse_args()

    model = load_model(Path(args.model))
    detections = analyze_audio(model, Path(args.audio))
    Path(args.output).write_text(json.dumps(detections, indent=2))
    print(f"Detected {len(detections)} events -> {args.output}")
