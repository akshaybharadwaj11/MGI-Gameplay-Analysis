#!/usr/bin/env python3
"""
Gameplay Event Detection — VLM-Based Pipeline
==============================================

Detects touchdown events in gridiron football gameplay footage using
Qwen2.5-VL (open-source vision-language model). No training required.
"""

import argparse
import os
import sys
import json
import csv
import time
import subprocess
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


# ============================================================
# Device Detection & Configuration
# ============================================================

def detect_device() -> Tuple[str, str, str]:
    """
    Auto-detect best device and appropriate model/dtype.

    Returns: (device, model_id, torch_dtype)
      - A100/H100 (≥30GB VRAM)  → Qwen2.5-VL-7B, float16
      - T4/V100 (<30GB VRAM)     → Qwen2.5-VL-3B, float16
      - Apple Silicon (MPS)      → SmolVLM2-2.2B, float16
      - CPU                      → SmolVLM2-500M, float32
    """
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"[device] CUDA: {name} ({vram_gb:.1f} GB, "
              f"compute {capability[0]}.{capability[1]})")

        if vram_gb >= 30:
            return "cuda", "Qwen/Qwen2.5-VL-7B-Instruct", "float16"
        else:
            return "cuda", "Qwen/Qwen2.5-VL-3B-Instruct", "float16"

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[device] Apple Silicon (MPS)")
        return "mps", "HuggingFaceTB/SmolVLM2-2.2B-Instruct", "float16"

    else:
        print("[device] CPU — using SmolVLM2-500M for fast inference")
        return "cpu", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "float32"


@dataclass
class Config:
    """All pipeline parameters in one place."""

    # ---- Video Sources (YouTube download) ----
    video_sources: List[dict] = field(default_factory=lambda: [
        {
            "name": "blitz_n64_longplay",
            "url": "https://www.youtube.com/watch?v=UqKErnm382M",
            "trim_start": 0,
            "trim_end": 600,
        },
        {
            "name": "blitz_ps3_hd",
            "url": "https://www.youtube.com/watch?v=CqJBdaU0MOs",
            "trim_start": 0,
            "trim_end": 600,
        },
    ])

    # ---- Local video files (passed via --input) ----
    local_videos: List[str] = field(default_factory=list)
    # Whether to also download YouTube sources
    do_download: bool = False

    data_dir: str = "data"
    output_dir: str = "outputs"

    # ---- Frame Sampling ----
    sample_interval: int = 15  # every 15th frame ≈ 2 fps
    frame_resize: Tuple[int, int] = (640, 480)

    # ---- VLM Model (auto-detected, override with --model) ----
    model_id: str = ""  # set by detect_device() or --model
    torch_dtype: str = ""  # set by detect_device()
    device: str = ""  # set by detect_device()
    max_new_tokens: int = 50
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 512 * 28 * 28

    # ---- Detection Prompt ----
    detection_prompt: str = (
        "You are analyzing a frame from an American football video game. "
        "Is this frame showing a TOUCHDOWN event? A touchdown is indicated by: "
        "the word 'TOUCHDOWN' displayed on screen, a scoring celebration, "
        "players in the end zone after scoring, or a post-score replay. "
        "Respond with EXACTLY one line in this format:\n"
        "TOUCHDOWN: YES or TOUCHDOWN: NO\n"
        "Then on the next line, CONFIDENCE: a number from 0.0 to 1.0\n"
        "Nothing else."
    )

    # ---- Post-Processing ----
    confidence_threshold: float = 0.5
    min_event_samples: int = 2
    merge_gap_samples: int = 4
    min_event_duration_sec: float = 1.5

    # ---- Output ----
    json_path: str = "outputs/events_output.json"
    csv_path: str = "outputs/events_output.csv"
    timeline_path: str = "outputs/detection_timeline.png"
    gifs_dir: str = "outputs/event_gifs"


# ============================================================
# Video Acquisition (download + local)
# ============================================================

def download_video(source: dict, data_dir: str) -> Optional[str]:
    """Download and trim a single video. Returns path or None on failure."""
    raw_dir = Path(data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{source['name']}.mp4"

    if output_path.exists():
        print(f"[download] {source['name']} already exists, skipping.")
        return str(output_path)

    url = source["url"]
    temp_path = raw_dir / f"{source['name']}.temp.mp4"

    print(f"[download] Downloading {source['name']} ...")
    try:
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", str(temp_path),
            "--no-playlist",
            "--quiet",
            url,
        ]
        subprocess.run(cmd, check=True, timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError) as e:
        print(f"[download] FAILED: {source['name']} — {e}")
        temp_path.unlink(missing_ok=True)
        return None

    # Trim if needed
    trim_start = source.get("trim_start", 0)
    trim_end = source.get("trim_end")
    if trim_end:
        duration = trim_end - trim_start
        print(f"[download] Trimming {source['name']}: {trim_start}s–{trim_end}s")
        try:
            trim_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(temp_path),
                "-ss", str(trim_start),
                "-t", str(duration),
                "-c", "copy",
                str(output_path),
            ]
            subprocess.run(trim_cmd, check=True, timeout=120)
            temp_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[download] Trim failed, using full video: {e}")
            temp_path.rename(output_path)
    else:
        temp_path.rename(output_path)

    print(f"[download] Saved → {output_path}")
    return str(output_path)


@dataclass
class VideoInput:
    """A video with optional trim range."""
    path: str
    start_frame: Optional[int] = None  # None = from beginning
    end_frame: Optional[int] = None    # None = to end


def _parse_input_spec(spec: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse input spec like:
        game.mp4              → (game.mp4, None, None)
        game.mp4[0:5400]      → (game.mp4, 0, 5400)       # frames
        game.mp4[10s:180s]    → needs fps, resolved later   # seconds
        game.mp4[0s:120s]     → needs fps, resolved later

    Returns (path, start, end) where start/end are frames or
    negative values indicating seconds (encoded as -seconds*1000 to
    distinguish from frame 0).
    """
    match = re.match(r'^(.+?)\[([^\]]+)\]$', spec)
    if not match:
        return spec, None, None

    path = match.group(1)
    range_str = match.group(2)

    parts = range_str.split(":")
    if len(parts) != 2:
        print(f"[input] Invalid trim range '{range_str}', expected start:end")
        return path, None, None

    def _parse_val(v: str) -> Optional[int]:
        v = v.strip()
        if not v:
            return None
        if v.lower().endswith("s"):
            # Seconds — encode as negative milliseconds
            secs = float(v[:-1])
            return -int(secs * 1000)
        return int(v)

    start = _parse_val(parts[0])
    end = _parse_val(parts[1])
    return path, start, end


def _resolve_trim_frames(
    start: Optional[int], end: Optional[int], fps: float
) -> Tuple[Optional[int], Optional[int]]:
    """Convert second-encoded values (negative ms) to frame numbers."""
    if start is not None and start < 0:
        start = int((-start / 1000) * fps)
    if end is not None and end < 0:
        end = int((-end / 1000) * fps)
    return start, end


def validate_local_video(spec: str) -> Optional[VideoInput]:
    """
    Parse input spec, validate file, return VideoInput.
    Supports trim syntax: game.mp4[0s:180s] or game.mp4[0:5400]
    """
    path_str, start, end = _parse_input_spec(spec)
    p = Path(path_str).resolve()

    if not p.exists():
        print(f"[input] File not found: {path_str}")
        return None

    supported = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}
    if p.suffix.lower() not in supported:
        print(f"[input] Unsupported format: {p.suffix} — "
              f"supported: {', '.join(sorted(supported))}")
        return None

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        print(f"[input] Cannot open video: {path_str}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0
    cap.release()

    # Resolve seconds to frames
    start, end = _resolve_trim_frames(start, end, fps)

    # Clamp
    if start is not None:
        start = max(0, min(start, total - 1))
    if end is not None:
        end = max(0, min(end, total))

    trim_info = ""
    if start is not None or end is not None:
        s = start or 0
        e = end or total
        trim_info = (f" [trim: frame {s}–{e}, "
                     f"{s/fps:.1f}s–{e/fps:.1f}s]")

    print(f"[input] {p.name}: {total} frames, {fps:.1f} fps, "
          f"{duration:.1f}s{trim_info}")
    return VideoInput(path=str(p), start_frame=start, end_frame=end)


def acquire_videos(cfg: Config) -> List[VideoInput]:
    """Collect all video inputs — local files + downloads."""
    inputs = []

    # Local files
    for spec in cfg.local_videos:
        vi = validate_local_video(spec)
        if vi:
            inputs.append(vi)

    # YouTube downloads
    if cfg.do_download:
        for source in cfg.video_sources:
            path = download_video(source, cfg.data_dir)
            if path:
                inputs.append(VideoInput(path=path))

    if not inputs:
        print("ERROR: No videos available.")
        print("  Pass local files:  --input game.mp4")
        print("  With trim:         --input game.mp4[0s:180s]")
        print("  Or download:       --download")
        sys.exit(1)

    print(f"[videos] {len(inputs)} video(s) ready.")
    return inputs


# ============================================================
# Frame Sampling
# ============================================================

@dataclass
class SampledFrame:
    """A single sampled frame with metadata."""
    video_name: str
    video_path: str
    frame_index: int
    timestamp_sec: float
    image_path: str


def sample_frames_from_video(
    video_input: VideoInput,
    cfg: Config,
) -> List[SampledFrame]:
    """Extract frames at regular intervals, respecting trim range."""
    video_path = video_input.path
    video_name = Path(video_path).stem
    frames_dir = Path(cfg.data_dir) / "frames" / video_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[frames] Cannot open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = cfg.sample_interval
    resize = cfg.frame_resize

    # Resolve trim range
    start_frame = video_input.start_frame or 0
    end_frame = video_input.end_frame or total

    # Check cache — but only if trim matches
    # Use a trim-aware cache key in filename
    cache_marker = frames_dir / f".trim_{start_frame}_{end_frame}"
    existing = sorted(frames_dir.glob("frame_*.jpg"))
    if existing and cache_marker.exists():
        samples = []
        for p in existing:
            idx = int(p.stem.split("_")[1])
            samples.append(SampledFrame(
                video_name=video_name,
                video_path=video_path,
                frame_index=idx,
                timestamp_sec=round(idx / fps, 3),
                image_path=str(p),
            ))
        print(f"[frames] {video_name}: {len(samples)} cached frames "
              f"(trim {start_frame}–{end_frame})")
        return samples

    # Clear old frames if re-extracting with different trim
    for f in existing:
        f.unlink()
    for m in frames_dir.glob(".trim_*"):
        m.unlink()

    trimmed_count = end_frame - start_frame
    sample_count = trimmed_count // interval
    print(f"[frames] {video_name}: sampling every {interval} frames "
          f"from frames {start_frame}–{end_frame} "
          f"({sample_count} samples, {start_frame/fps:.1f}s–{end_frame/fps:.1f}s)")

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    samples = []
    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % interval == 0:
            resized = cv2.resize(frame, resize)
            path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(path), resized)
            samples.append(SampledFrame(
                video_name=video_name,
                video_path=video_path,
                frame_index=frame_idx,
                timestamp_sec=round(frame_idx / fps, 3),
                image_path=str(path),
            ))
        frame_idx += 1
    cap.release()

    # Write cache marker
    cache_marker.touch()

    print(f"[frames] {video_name}: {len(samples)} frames sampled")
    return samples


def sample_all_frames(video_inputs: List[VideoInput], cfg: Config) -> List[SampledFrame]:
    """Sample frames from all videos."""
    all_samples = []
    for vi in video_inputs:
        samples = sample_frames_from_video(vi, cfg)
        all_samples.extend(samples)
    print(f"[frames] Total: {len(all_samples)} sampled frames across "
          f"{len(video_inputs)} video(s)")
    return all_samples


# ============================================================
# Load VLM
# ============================================================

def _is_smolvlm(model_id: str) -> bool:
    """Check if model_id is a SmolVLM variant."""
    return "SmolVLM" in model_id or "smolvlm" in model_id.lower()


def load_vlm(cfg: Config):
    """
    Load VLM with device-appropriate settings.
    Supports both Qwen2.5-VL and SmolVLM2 model families.
    Returns (model, processor).
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    dtype = getattr(torch, cfg.torch_dtype)
    print(f"[vlm] Loading {cfg.model_id} on {cfg.device} ({cfg.torch_dtype}) ...")

    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[vlm] VRAM: {mem:.1f} GB")

    load_kwargs = dict(torch_dtype=dtype)

    # Attention implementation
    if cfg.device == "cuda":
        capability = torch.cuda.get_device_capability(0)
        if capability[0] >= 8:
            try:
                import flash_attn  # noqa: F401
                load_kwargs["attn_implementation"] = "flash_attention_2"
                print(f"[vlm] Using flash_attention_2 "
                      f"(compute {capability[0]}.{capability[1]})")
            except ImportError:
                print("[vlm] flash-attn not installed, using eager/SDPA")
        load_kwargs["device_map"] = "auto"
    elif cfg.device == "mps":
        load_kwargs["device_map"] = None
    else:
        # CPU — use eager attention (SDPA can be slow on CPU)
        load_kwargs["_attn_implementation"] = "eager"
        load_kwargs["device_map"] = None

    # Load model — use AutoModelForImageTextToText which works for both
    # Qwen2.5-VL and SmolVLM2 families
    if _is_smolvlm(cfg.model_id):
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.model_id, **load_kwargs
        )
    else:
        # Qwen models — can also use AutoModelForImageTextToText but
        # the dedicated class is more reliable
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cfg.model_id, **load_kwargs
            )
        except (ImportError, Exception):
            model = AutoModelForImageTextToText.from_pretrained(
                cfg.model_id, **load_kwargs
            )

    # Move to MPS if needed
    if cfg.device == "mps":
        model = model.to("mps")

    model.eval()

    # Processor
    proc_kwargs = {}
    if not _is_smolvlm(cfg.model_id):
        # Qwen models support min/max_pixels
        proc_kwargs["min_pixels"] = cfg.min_pixels
        proc_kwargs["max_pixels"] = cfg.max_pixels

    processor = AutoProcessor.from_pretrained(cfg.model_id, **proc_kwargs)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[vlm] Loaded: {param_count:.0f}M parameters on {cfg.device}")
    return model, processor


# ============================================================
# VLM Inference
# ============================================================

@dataclass
class FramePrediction:
    """Prediction result for a single frame."""
    video_name: str
    frame_index: int
    timestamp_sec: float
    is_touchdown: bool
    confidence: float
    raw_response: str


def _parse_vlm_response(text: str) -> Tuple[bool, float]:
    """Parse VLM response to extract touchdown label and confidence."""
    text_upper = text.upper().strip()

    is_td = False
    if "TOUCHDOWN: YES" in text_upper or "TOUCHDOWN:YES" in text_upper:
        is_td = True
    elif "YES" in text_upper and "NO" not in text_upper:
        is_td = True

    confidence = 0.8 if is_td else 0.2
    conf_match = re.search(r"CONFIDENCE[:\s]*([0-9]*\.?[0-9]+)", text_upper)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            pass

    return is_td, confidence


def run_vlm_inference(
    samples: List[SampledFrame],
    model,
    processor,
    cfg: Config,
) -> List[FramePrediction]:
    """Run VLM on each sampled frame. Returns per-frame predictions."""
    device = cfg.device
    predictions = []

    # Check cache
    cache_path = Path(cfg.output_dir) / "predictions_cache.json"
    if cache_path.exists():
        print(f"[vlm] Loading cached predictions from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        for c in cached:
            predictions.append(FramePrediction(**c))
        if len(predictions) == len(samples):
            return predictions
        else:
            print(f"[vlm] Cache mismatch ({len(predictions)} vs {len(samples)}), "
                  "re-running.")
            predictions = []

    total = len(samples)
    t0 = time.time()
    save_interval = 50

    # Estimate speed by device and model size
    if _is_smolvlm(cfg.model_id):
        speed_est = {"cuda": 0.3, "mps": 1.5, "cpu": 3.0}.get(device, 3.0)
    else:
        speed_est = {"cuda": 1.0, "mps": 3.0, "cpu": 10.0}.get(device, 5.0)
    print(f"[vlm] Running inference on {total} frames ...")
    print(f"[vlm] Estimated time: ~{total * speed_est / 60:.0f} min "
          f"(~{speed_est:.0f}s/frame on {device})")

    smolvlm = _is_smolvlm(cfg.model_id)

    for i, sample in enumerate(samples):
        image = Image.open(sample.image_path).convert("RGB")

        if smolvlm:
            # SmolVLM2 format: {"type": "image"} placeholder, image passed separately
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": cfg.detection_prompt},
                    ],
                }
            ]
        else:
            # Qwen format: image object inline
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": cfg.detection_prompt},
                    ],
                }
            ]

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to correct device
        if device == "cuda":
            inputs = inputs.to("cuda")
        elif device == "mps":
            inputs = inputs.to("mps")
        # CPU: already there

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        is_td, confidence = _parse_vlm_response(response)

        pred = FramePrediction(
            video_name=sample.video_name,
            frame_index=sample.frame_index,
            timestamp_sec=sample.timestamp_sec,
            is_touchdown=is_td,
            confidence=confidence,
            raw_response=response.strip(),
        )
        predictions.append(pred)

        if (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            td_count = sum(1 for p in predictions if p.is_touchdown)
            print(f"  [{i+1}/{total}] {rate:.1f} frames/sec | "
                  f"ETA: {eta / 60:.1f} min | TDs found: {td_count}")

        if (i + 1) % save_interval == 0:
            _save_predictions_cache(predictions, cache_path)

        del inputs, output_ids, generated_ids
        if device == "cuda":
            torch.cuda.empty_cache()

    _save_predictions_cache(predictions, cache_path)

    elapsed = time.time() - t0
    td_count = sum(1 for p in predictions if p.is_touchdown)
    print(f"[vlm] Done: {total} frames in {elapsed:.0f}s "
          f"({elapsed/total:.1f}s/frame), {td_count} TD frames")

    return predictions


def _save_predictions_cache(predictions: List[FramePrediction], path: Path):
    """Save predictions to JSON cache for fault tolerance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(p) for p in predictions], f, indent=2)


# ============================================================
# Post-Processing — Predictions → Events
# ============================================================

@dataclass
class DetectedEvent:
    """A single detected touchdown event."""
    event_id: int
    video_name: str
    start_frame: int
    end_frame: int
    start_timestamp_sec: float
    end_timestamp_sec: float
    duration_sec: float
    avg_confidence: float
    peak_confidence: float
    num_positive_frames: int
    detection_method: str = ""
    gif_path: str = ""


def predictions_to_events(
    predictions: List[FramePrediction],
    cfg: Config,
) -> List[DetectedEvent]:
    """Convert per-frame predictions into discrete touchdown events."""
    by_video = {}
    for p in predictions:
        by_video.setdefault(p.video_name, []).append(p)

    all_events = []
    event_counter = 0
    model_short = cfg.model_id.split("/")[-1]

    for video_name, preds in by_video.items():
        preds.sort(key=lambda p: p.frame_index)

        positive = [
            p for p in preds
            if p.is_touchdown and p.confidence >= cfg.confidence_threshold
        ]
        if not positive:
            continue

        # Find contiguous runs
        runs = []
        current_run = [positive[0]]
        for p in positive[1:]:
            prev = current_run[-1]
            frame_gap = (p.frame_index - prev.frame_index) / cfg.sample_interval
            if frame_gap <= cfg.merge_gap_samples:
                current_run.append(p)
            else:
                runs.append(current_run)
                current_run = [p]
        runs.append(current_run)

        # Filter by duration and min samples
        for run in runs:
            if len(run) < cfg.min_event_samples:
                continue

            duration = run[-1].timestamp_sec - run[0].timestamp_sec
            if duration < cfg.min_event_duration_sec:
                continue

            event_counter += 1
            confidences = [p.confidence for p in run]

            event = DetectedEvent(
                event_id=event_counter,
                video_name=video_name,
                start_frame=run[0].frame_index,
                end_frame=run[-1].frame_index,
                start_timestamp_sec=run[0].timestamp_sec,
                end_timestamp_sec=run[-1].timestamp_sec,
                duration_sec=round(duration, 3),
                avg_confidence=round(float(np.mean(confidences)), 4),
                peak_confidence=round(float(np.max(confidences)), 4),
                num_positive_frames=len(run),
                detection_method=f"{model_short} zero-shot",
            )
            all_events.append(event)

    print(f"[events] {len(all_events)} event(s) detected "
          f"(min duration: {cfg.min_event_duration_sec}s)")
    return all_events


# ============================================================
# Output — JSON + CSV + GIFs + Timeline
# ============================================================

def build_output(
    events: List[DetectedEvent],
    predictions: List[FramePrediction],
    video_inputs: List[VideoInput],
    cfg: Config,
) -> dict:
    """Build and save all output files."""
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    video_metadata = []
    for vi in video_inputs:
        cap = cv2.VideoCapture(vi.path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        sf = vi.start_frame or 0
        ef = vi.end_frame or total
        video_metadata.append({
            "filename": Path(vi.path).name,
            "source_path": vi.path,
            "fps": round(fps, 2),
            "total_frames": total,
            "duration_seconds": round(total / fps, 2),
            "trim_start_frame": sf,
            "trim_end_frame": ef,
            "trim_duration_seconds": round((ef - sf) / fps, 2),
        })

    total_sampled = len(predictions)
    td_frames = sum(1 for p in predictions if p.is_touchdown)

    output = {
        "pipeline": {
            "method": "VLM zero-shot inference (no training)",
            "model": cfg.model_id,
            "device": cfg.device,
            "sample_interval": cfg.sample_interval,
            "confidence_threshold": cfg.confidence_threshold,
            "min_event_duration_sec": cfg.min_event_duration_sec,
        },
        "video_metadata": video_metadata,
        "summary": {
            "total_frames_sampled": total_sampled,
            "touchdown_frames": td_frames,
            "touchdown_frame_rate": round(td_frames / max(total_sampled, 1), 4),
            "total_events_detected": len(events),
        },
        "events": [asdict(e) for e in events],
    }

    # JSON
    json_path = Path(cfg.json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[output] JSON → {json_path}")

    # CSV
    csv_path = Path(cfg.csv_path)
    if events:
        fieldnames = list(asdict(events[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in events:
                writer.writerow(asdict(e))
        print(f"[output] CSV → {csv_path}")

    # Generate GIFs
    video_paths = [vi.path for vi in video_inputs]
    _generate_event_gifs(events, video_paths, cfg)

    # Timeline plot
    _plot_timeline(predictions, events, cfg)

    return output


def _generate_event_gifs(
    events: List[DetectedEvent],
    video_paths: List[str],
    cfg: Config,
):
    """Extract frames for each event from original video → animated GIF."""
    gifs_dir = Path(cfg.gifs_dir)
    gifs_dir.mkdir(parents=True, exist_ok=True)

    vpath_map = {Path(vp).stem: vp for vp in video_paths}

    for event in events:
        vp = vpath_map.get(event.video_name)
        if not vp:
            print(f"[gif] Video not found for {event.video_name}, skipping.")
            continue

        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # 0.5s padding on each side for context
        pad = int(fps * 0.5)
        start = max(0, event.start_frame - pad)
        end = event.end_frame + pad

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        pil_frames = []

        for _ in range(start, end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb).resize((480, 360)))

        cap.release()

        if not pil_frames:
            continue

        gif_fps = min(fps, 15)
        gif_path = gifs_dir / f"event_{event.event_id}_{event.video_name}.gif"
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / gif_fps),
            loop=0,
            optimize=True,
        )
        event.gif_path = str(gif_path)
        print(f"[gif] Event #{event.event_id}: {len(pil_frames)} frames → {gif_path}")

    print(f"[gif] Generated {sum(1 for e in events if e.gif_path)} GIF(s)")


def _plot_timeline(
    predictions: List[FramePrediction],
    events: List[DetectedEvent],
    cfg: Config,
):
    """Plot detection confidence over time per video."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[output] matplotlib not available, skipping plot.")
        return

    by_video = {}
    for p in predictions:
        by_video.setdefault(p.video_name, []).append(p)

    n = len(by_video)
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), squeeze=False)

    for idx, (vname, preds) in enumerate(by_video.items()):
        ax = axes[idx, 0]
        preds.sort(key=lambda p: p.timestamp_sec)

        ts = [p.timestamp_sec for p in preds]
        conf = [p.confidence if p.is_touchdown else 0.0 for p in preds]

        ax.fill_between(ts, conf, alpha=0.3, color="red", label="TD confidence")
        ax.axhline(y=cfg.confidence_threshold, color="gray", linestyle="--",
                     alpha=0.5, label=f"Threshold ({cfg.confidence_threshold})")

        video_events = [e for e in events if e.video_name == vname]
        for e in video_events:
            ax.axvspan(e.start_timestamp_sec, e.end_timestamp_sec,
                        alpha=0.2, color="green",
                        label="Event" if e == video_events[0] else "")

        ax.set_title(f"{vname} — {len(video_events)} event(s)")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Confidence")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(Path(cfg.timeline_path), dpi=150)
    plt.close()
    print(f"[output] Timeline → {cfg.timeline_path}")


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(cfg: Config):
    """Execute the full detection pipeline."""
    print("=" * 60)
    print("  Gameplay Event Detection — VLM Pipeline")
    print(f"  Model:  {cfg.model_id}")
    print(f"  Device: {cfg.device} ({cfg.torch_dtype})")
    print("=" * 60)

    # Stage 1: Acquire videos
    print("\n" + "=" * 60)
    print("  STAGE 1: Video Acquisition")
    print("=" * 60)
    video_inputs = acquire_videos(cfg)

    # Stage 2: Sample frames
    print("\n" + "=" * 60)
    print("  STAGE 2: Sample Frames")
    print("=" * 60)
    samples = sample_all_frames(video_inputs, cfg)

    # Stage 2: Sample frames
    print("\n" + "=" * 60)
    print("  STAGE 2: Sample Frames")
    print("=" * 60)
    samples = sample_all_frames(video_inputs, cfg)
    if not samples:
        print("ERROR: No frames sampled.")
        return

    # Stage 3: Load VLM
    print("\n" + "=" * 60)
    print("  STAGE 3: Load VLM")
    print("=" * 60)
    model, processor = load_vlm(cfg)

    # Stage 4: Inference
    print("\n" + "=" * 60)
    print("  STAGE 4: VLM Inference")
    print("=" * 60)
    predictions = run_vlm_inference(samples, model, processor, cfg)

    # Free memory
    del model, processor
    if cfg.device == "cuda":
        torch.cuda.empty_cache()

    # Stage 5: Post-process
    print("\n" + "=" * 60)
    print("  STAGE 5: Post-Processing → Events")
    print("=" * 60)
    events = predictions_to_events(predictions, cfg)

    # Stage 6: Output
    print("\n" + "=" * 60)
    print("  STAGE 6: Output (JSON + CSV + GIFs)")
    print("=" * 60)
    output = build_output(events, predictions, video_inputs, cfg)

    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Videos:  {len(video_inputs)}")
    print(f"  Frames:  {len(samples)}")
    print(f"  Events:  {len(events)}")
    for e in events:
        gif = f" → {e.gif_path}" if e.gif_path else ""
        print(f"    #{e.event_id} [{e.video_name}] "
              f"{e.start_timestamp_sec:.1f}s–{e.end_timestamp_sec:.1f}s "
              f"({e.duration_sec:.1f}s, conf={e.avg_confidence:.2f}){gif}")
    print(f"\n  Outputs:")
    print(f"    JSON: {cfg.json_path}")
    print(f"    CSV:  {cfg.csv_path}")
    print(f"    GIFs: {cfg.gifs_dir}/")
    print(f"    Plot: {cfg.timeline_path}")

    return output


# ============================================================
# CLI & Entry Point
# ============================================================

def parse_args() -> Config:
    """Parse CLI arguments and build Config."""
    parser = argparse.ArgumentParser(
        description="Detect touchdown events in football gameplay video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gameplay_event_detection.py --input game.mp4
  python gameplay_event_detection.py --input clip1.mov clip2.mp4
  python gameplay_event_detection.py --input recordings/*.mp4
  python gameplay_event_detection.py --download

  Trim by frame range:
  python gameplay_event_detection.py --input game.mp4[0:5400]

  Trim by seconds (use 's' suffix):
  python gameplay_event_detection.py --input game.mp4[0s:180s]

  Mixed trims:
  python gameplay_event_detection.py --input game1.mp4[0s:120s] game2.mov[60s:300s] game3.mp4

  Model selection:
  python gameplay_event_detection.py --input game.mp4 --model 7b     # A100
  python gameplay_event_detection.py --input game.mp4 --model 500m   # CPU
        """,
    )
    parser.add_argument(
        "--input", "-i", nargs="+", default=[],
        help="Local video file(s): .mp4, .mov, .avi, .mkv, .webm, .m4v",
    )
    parser.add_argument(
        "--download", "-d", action="store_true",
        help="Download configured YouTube videos",
    )
    parser.add_argument(
        "--model", "-m",
        choices=["auto", "7b", "3b", "2b", "500m", "256m"],
        default="auto",
        help="Model size: 7b/3b (Qwen, GPU), 2b/500m/256m (SmolVLM2, CPU-friendly). "
             "Default: auto-detect from hardware.",
    )
    parser.add_argument(
        "--sample-interval", type=int, default=None,
        help="Sample every N-th frame (default: 15 ≈ 2fps)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--min-duration", type=float, default=None,
        help="Minimum event duration in seconds (default: 1.5)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: outputs/)",
    )

    args = parser.parse_args()

    # If no input and no download, default to download
    if not args.input and not args.download:
        print("[main] No --input or --download specified. Defaulting to --download.")
        args.download = True

    # Auto-detect device and model
    device, auto_model, auto_dtype = detect_device()

    # Model override
    model_map = {
        "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "500m": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "256m": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    }
    if args.model != "auto":
        model_id = model_map[args.model]
        # Warn if large model on small GPU
        if args.model == "7b" and device == "cuda":
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram < 30:
                print(f"[warn] 7B requested but only {vram:.0f}GB VRAM. "
                      "May OOM — consider --model 3b or --model 500m")
        dtype = "float32" if device == "cpu" else "float16"
    else:
        model_id = auto_model
        dtype = auto_dtype

    cfg = Config(
        local_videos=args.input,
        do_download=args.download,
        model_id=model_id,
        torch_dtype=dtype,
        device=device,
    )

    if args.sample_interval is not None:
        cfg.sample_interval = args.sample_interval
    if args.threshold is not None:
        cfg.confidence_threshold = args.threshold
    if args.min_duration is not None:
        cfg.min_event_duration_sec = args.min_duration
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
        cfg.json_path = f"{args.output_dir}/events_output.json"
        cfg.csv_path = f"{args.output_dir}/events_output.csv"
        cfg.timeline_path = f"{args.output_dir}/detection_timeline.png"
        cfg.gifs_dir = f"{args.output_dir}/event_gifs"

    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    output = run_pipeline(cfg)