import argparse
from pathlib import Path
import tempfile
import csv

import librosa
import numpy as np
import torch
from tqdm import tqdm

from audio_labeling_tools import run_ffmpeg_extract, detect_onsets
from acoustic_model import load_acoustic_model

# video processing modules
from video_models.ball_detector import BallDetector
from video_models.court_detection_net import CourtDetectorNet
from video_models.person_detector import PersonDetector
from video_models.bounce_detector import BounceDetector

def predict_hits(audio_path: Path, model_path: Path) -> list[float]:
    y, sr = librosa.load(str(audio_path), sr=48_000, mono=True)
    onsets = detect_onsets(y, sr, hop_len=int(sr*0.025), delta=0.2, min_gap_sec=0.3)
    model = load_acoustic_model(model_path)
    pre_ms = 200
    post_ms = 200
    hits = []
    for s in onsets:
        start = max(0, s - int(pre_ms*sr/1000))
        end = min(len(y), s + int(post_ms*sr/1000))
        clip = y[start:end]
        mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=20)
        x = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            prob = torch.softmax(model(x), dim=1)[0, 0].item()
        if prob > 0.5:
            hits.append(start/sr)
    return hits


def process_video(video_path: Path, models_dir: Path):
    frames, fps = read_video(video_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ball = BallDetector(models_dir / "ball_track_model.pt", device)
    court = CourtDetectorNet(models_dir / "court_model.pt", device)
    bounce = BounceDetector(models_dir / "bounce_model.cbm")
    persons = PersonDetector(device=device)

    scenes = scene_detect(str(video_path))
    ball_track = ball.infer_model(frames)
    matrices, kps = court.infer_model(frames)
    persons_top, persons_bottom = persons.track_players(frames, matrices, filter_players=False)
    x_ball = [b[0] for b in ball_track]
    y_ball = [b[1] for b in ball_track]
    bounces = bounce.predict(x_ball, y_ball)

    return {
        "fps": fps,
        "ball_track": ball_track,
        "matrices": matrices,
        "kps": kps,
        "persons_top": persons_top,
        "persons_bottom": persons_bottom,
        "bounces": bounces,
        "scenes": scenes,
    }


def read_video(path_video: Path):
    import cv2
    cap = cv2.VideoCapture(str(path_video))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps

from video_models.utils import scene_detect


def analyze(video_path: Path, models_dir: Path, output_csv: Path):
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = Path(tmp)/"audio.wav"
        run_ffmpeg_extract(video_path, audio_path)
        hits = predict_hits(audio_path, models_dir / "acoustic_classifier.pth")

    video_data = process_video(video_path, models_dir)

    rows = []
    for t in hits:
        sec = int(t*video_data["fps"])
        hit_frame = min(sec, len(video_data["ball_track"])-1)
        who = "A" if video_data["persons_bottom"][hit_frame] else "B"
        rows.append({"time": t, "player": who})

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "player"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tennis match analyzer")
    p.add_argument("input_video", type=Path)
    p.add_argument("--models_dir", type=Path, default=Path("models"))
    p.add_argument("--output_csv", type=Path, default=Path("results.csv"))
    args = p.parse_args()
    analyze(args.input_video, args.models_dir, args.output_csv)
