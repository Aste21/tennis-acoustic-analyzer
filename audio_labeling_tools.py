"""
audio_labeling_tools.py — prepare audio (and optional video) clips
==================================================================

* extract : pull mono 48 kHz WAV from a video (ffmpeg)
* slice   : detect peaks/onsets, cut fixed-length WAV (and MP4) clips,
            and write a CSV ready for labelling.

run quick_prepare.ps1 for easy launch of the application. 
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import List

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ───────────────────────── ffmpeg helpers ────────────────────────────────
def run_ffmpeg_extract(
    input_path: Path, output_path: Path, sr: int = 48_000, mono: bool = True
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file {output_path}")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ac", "1" if mono else "2",
            "-ar", str(sr),
            str(output_path),
        ],
        check=True,
    )


def ffmpeg_slice_video(
    src: Path,
    dst: Path,
    start_sec: float,
    duration_sec: float,
    codec: str = "libx264",
    scale: str | None = "-1:360",
) -> None:
    vf = f"scale={scale}" if scale else "null"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}", "-t", f"{duration_sec:.3f}",
            "-i", str(src),
            "-vf", vf,
            "-c:v", codec,
            "-c:a", "aac", "-ac", "1", "-ar", "48000",
            "-loglevel", "error",
            str(dst),
        ],
        check=True,
    )

# ─────────────────────── onset-detector helper ───────────────────────────
def detect_onsets(
    y: np.ndarray,
    sr: int,
    hop_len: int,
    delta: float,
    min_gap_sec: float,
) -> List[int]:
    """Return sample indices of spectral-flux onsets spaced ≥ *min_gap_sec*."""
    onsets = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_len,
        backtrack=True,
        pre_max=20, post_max=20,
        pre_avg=100, post_avg=100,
        delta=delta,
        units="samples",
    )
    gap = int(min_gap_sec * sr)
    filtered: List[int] = []
    last = -gap
    for s in sorted(onsets):
        if s - last >= gap:
            filtered.append(s)
            last = s
    return filtered

# ───────────────────────── CLI handlers ──────────────────────────────────
def cmd_extract(args: argparse.Namespace) -> None:
    run_ffmpeg_extract(
        Path(args.input).expanduser(),
        Path(args.output).expanduser(),
        sr=args.sr,
        mono=not args.stereo,
    )
    print("✅ WAV extracted")


def cmd_slice(args: argparse.Namespace) -> None:
    in_path  = Path(args.input)
    out_dir  = Path(args.output_dir)
    csv_path = Path(args.csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # ── load WAV ─────────────────────────────────────────────────────────
    y, sr_native = librosa.load(in_path, sr=None, mono=True)
    sr = args.sr or sr_native
    if sr != sr_native:
        y = librosa.resample(y, sr_native, sr)

    hop = int(sr * args.hop_ms / 1000)
    win = int(sr * args.win_ms / 1000)

    # spacing between clips
    gap_ms = args.min_gap_ms if args.min_gap_ms is not None else (args.pre_ms + args.post_ms)
    min_gap_sec = gap_ms / 1000

    # ── detect peaks / onsets ────────────────────────────────────────────
    if args.method == "rms":
        env = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True)[0]
        env_norm = env / (env.max() or 1.0)
        wait_frames = max(3, int(gap_ms / args.hop_ms))
        peaks = librosa.util.peak_pick(
            env_norm,
            pre_max=3, post_max=3,
            pre_avg=3, post_avg=3,
            delta=args.threshold,
            wait=wait_frames,
        )
        sample_peaks = [int(p * hop + win / 2) for p in peaks]
    else:  # onset
        sample_peaks = detect_onsets(
            y, sr,
            hop_len=hop,
            delta=args.delta,
            min_gap_sec=min_gap_sec,
        )

    if not sample_peaks:
        print("No peaks detected – try a lower --delta or --threshold.")
        return

    # ── slice around peaks ───────────────────────────────────────────────
    pre  = int(args.pre_ms  * sr / 1000)
    post = int(args.post_ms * sr / 1000)
    duration_sec = (pre + post) / sr

    csv_rows = []
    video_src = Path(args.video) if (args.export_video and args.video) else None

    print(f"Slicing {len(sample_peaks)} clips …")
    for idx, p in tqdm(list(enumerate(sample_peaks, 1)), total=len(sample_peaks)):
        start = max(p - pre, 0)
        end   = min(p + post, len(y))
        clip  = y[start:end]

        wav_name = f"clip_{idx:05d}.wav"
        sf.write(out_dir / wav_name, clip, sr)

        clip_start_sec = start / sr
        video_name = ""
        if video_src:
            video_name = wav_name.replace(".wav", ".mp4")
            ffmpeg_slice_video(
                src=video_src,
                dst=out_dir / video_name,
                start_sec=clip_start_sec,
                duration_sec=duration_sec,
                codec=args.video_codec,
                scale=args.scale,
            )

        csv_rows.append([wav_name, p, f"{clip_start_sec:.3f}", "", video_name])

    # ── write CSV ────────────────────────────────────────────────────────
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "peak_sample", "clip_start_sec", "label", "video_id"])
        writer.writerows(csv_rows)

    print(f"✅ {len(csv_rows)} clips + CSV written to {out_dir}")

# ───────────────────────── argument parser ───────────────────────────────
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audio (and optional video) clip slicer")
    sub = p.add_subparsers(dest="command", required=True)

    # extract
    ext = sub.add_parser("extract", help="Extract mono WAV from a video")
    ext.add_argument("--input", required=True)
    ext.add_argument("--output", required=True)
    ext.add_argument("--sr", type=int, default=48_000)
    ext.add_argument("--stereo", action="store_true")
    ext.set_defaults(func=cmd_extract)

    # slice
    slc = sub.add_parser("slice", help="Detect peaks/onsets and cut clips")
    slc.add_argument("--input", required=True)
    slc.add_argument("--output_dir", required=True)
    slc.add_argument("--csv", required=True)

    # window + detection
    slc.add_argument("--method", choices=["rms", "onset"], default="onset",
                     help="Detection method (default: onset)")
    slc.add_argument("--delta", type=float, default=0.2,
                     help="Spectral-flux onset threshold (lower = more peaks)")
    slc.add_argument("--threshold", type=float, default=0.35,
                     help="RMS peak threshold (used only when --method rms)")
    slc.add_argument("--min_gap_ms", type=float,
                     help="Minimum spacing between clips in ms "
                          "(default = pre_ms + post_ms)")
    slc.add_argument("--sr", type=int, default=0)
    slc.add_argument("--hop_ms",  type=float, default=25.0)
    slc.add_argument("--win_ms",  type=float, default=50.0)
    slc.add_argument("--pre_ms",  type=float, default=200.0)
    slc.add_argument("--post_ms", type=float, default=200.0)

    # video
    slc.add_argument("--video", help="Original full-length video file")
    slc.add_argument("--export_video", action="store_true", help="Write MP4 clip too")
    slc.add_argument("--video_codec", default="libx264")
    slc.add_argument("--scale", default="-1:360")

    slc.set_defaults(func=cmd_slice)
    return p

# ───────────────────────── entry-point ────────────────────────────────────
if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    args.func(args)
