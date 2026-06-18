#!/usr/bin/env python3
"""
Yard Line Detection — Hybrid Classical-CV + VLM Pipeline
=========================================================

Detects yard lines in gridiron football gameplay video (NFL Blitz and similar)
and assigns each detected line a yard-marker value.

Two-phase hybrid design:

  Phase 1 — Classical CV (every sampled frame, fast):
    HSV field mask -> white line-pixel extraction -> morphological cleanup ->
    Hough line transform -> vanishing-point RANSAC consensus -> even-spacing
    filter. Produces ordered yard-line pixel positions per frame.

  Phase 2 — VLM yard-number reading (periodic, slow but semantic):
    A vision-language model (Qwen2.5-VL / SmolVLM2) reads the painted yard
    numbers off the field every `vlm_interval` sampled frames.

  Phase 3 — Fusion:
    Anchor CV pixel positions to VLM yard values, then propagate the anchor
    across intermediate frames using CV line geometry.

Outputs: per-frame JSON, aggregated yard-range events (JSON + CSV), annotated
JPEG frames (red line overlays + yard labels), and a timeline plot.

Mirrors the structure/conventions of gameplay_event_detection.py.
"""

import argparse
import os
import sys
import json
import csv
import time
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import torch
from PIL import Image


# ============================================================
# Device Detection & Configuration
# ============================================================

def detect_device() -> Tuple[str, str, str]:
    """
    Auto-detect best device and an appropriate VLM model/dtype.

    Returns: (device, model_id, torch_dtype)
      - A100/H100 (>=30GB VRAM)  -> Qwen2.5-VL-7B, float16
      - T4/V100 (<30GB VRAM)     -> Qwen2.5-VL-3B, float16
      - Apple Silicon (MPS)      -> SmolVLM2-2.2B, float16
      - CPU                      -> SmolVLM2-500M, float32
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

    # ---- Local video files (passed via --input) ----
    local_videos: List[str] = field(default_factory=list)

    data_dir: str = "data"
    output_dir: str = "outputs_yards"

    # ---- Frame Sampling ----
    sample_interval: int = 5          # CV runs on every 5th frame (~6 fps @ 30fps)
    frame_resize: Tuple[int, int] = (640, 480)

    # ---- VLM Model (auto-detected, override with --model) ----
    model_id: str = ""                # set by detect_device() or --model
    torch_dtype: str = ""             # set by detect_device()
    device: str = ""                  # set by detect_device()
    max_new_tokens: int = 64
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 512 * 28 * 28
    vlm_interval: int = 12            # run VLM every 12th sampled frame (~2s)

    # ---- VLM Prompt ----
    detection_prompt: str = (
        "You are analyzing a frame from an American football video game. "
        "What yard line numbers are painted on the field in this image? "
        "List them left to right. "
        "Respond with EXACTLY one line in this format:\n"
        "YARD_LINES: 20, 25, 30, 35\n"
        "If no yard numbers are visible, respond with:\n"
        "YARD_LINES: NONE\n"
        "Nothing else."
    )

    # ---- Phase 1: Classical CV — Field Mask (HSV green) ----
    hsv_green_lower: Tuple[int, int, int] = (35, 40, 40)
    hsv_green_upper: Tuple[int, int, int] = (85, 255, 255)
    min_green_fraction: float = 0.02   # skip frame if < 2% of pixels are field

    # ---- Phase 1: Line-pixel extraction & morphology ----
    white_thresh: int = 180            # grayscale brightness for line pixels
    morph_h_kernel: Tuple[int, int] = (25, 1)   # horizontal kernel (erode verticals)
    morph_iterations: int = 1

    # ---- Phase 1: Hough ----
    hough_threshold: int = 50
    hough_min_line_len: int = 40
    hough_max_line_gap: int = 20
    max_line_angle_deg: float = 80.0   # reject near-vertical segments

    # ---- Phase 1: Vanishing-point RANSAC ----
    vp_cluster_eps: float = 50.0       # px radius for intersection clustering
    vp_inlier_tol_deg: float = 6.0     # angular tolerance toward VP
    min_lines_for_vp: int = 3

    # ---- Phase 1: Even-spacing filter ----
    spacing_tolerance: float = 0.4     # allowed gap-ratio deviation
    line_merge_px: float = 18.0        # merge near-duplicate lines (by y_center)

    # ---- Output paths (resolved against output_dir in parse_args) ----
    per_frame_json: str = "outputs_yards/per_frame_yard_lines.json"
    events_json: str = "outputs_yards/yard_events.json"
    events_csv: str = "outputs_yards/yard_events.csv"
    annotated_dir: str = "outputs_yards/annotated"
    timeline_path: str = "outputs_yards/yard_timeline.png"


# ============================================================
# Video Acquisition (local files + trim parsing)
# ============================================================

@dataclass
class VideoInput:
    """A video with optional trim range (in frames)."""
    path: str
    start_frame: Optional[int] = None   # None = from beginning
    end_frame: Optional[int] = None     # None = to end


def _parse_input_spec(spec: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse an input spec with optional trim range:
        game.mp4              -> (game.mp4, None, None)
        game.mp4[0:5400]      -> (game.mp4, 0, 5400)        # frames
        game.mp4[10s:180s]    -> resolved to frames later   # seconds

    Seconds are encoded as negative milliseconds (-int(secs*1000)) so a single
    Optional[int] field can mean "frame N" (>=0) or "N seconds, resolve later"
    (<0). _resolve_trim_frames() converts them once fps is known.
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
    Parse an input spec, validate the file, return a VideoInput.
    Supports trim syntax: game.mp4[0s:180s] or game.mp4[0:5400].
    Returns None (with a printed reason) if the file is unusable.
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

    start, end = _resolve_trim_frames(start, end, fps)

    if start is not None:
        start = max(0, min(start, total - 1))
    if end is not None:
        end = max(0, min(end, total))

    trim_info = ""
    if start is not None or end is not None:
        s = start or 0
        e = end or total
        trim_info = f" [trim: frame {s}–{e}, {s/fps:.1f}s–{e/fps:.1f}s]"

    print(f"[input] {p.name}: {total} frames, {fps:.1f} fps, "
          f"{duration:.1f}s{trim_info}")
    return VideoInput(path=str(p), start_frame=start, end_frame=end)


def acquire_videos(cfg: Config) -> List[VideoInput]:
    """Collect and validate all local video inputs."""
    inputs = []
    for spec in cfg.local_videos:
        vi = validate_local_video(spec)
        if vi:
            inputs.append(vi)

    if not inputs:
        print("ERROR: No videos available.")
        print("  Pass local files:  --input game.mp4")
        print("  With trim:         --input game.mp4[0s:180s]")
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
    sample_index: int       # 0-based index within this video's sampled sequence
    frame_index: int        # original frame index in the video
    timestamp_sec: float
    image_path: str


def sample_frames_from_video(
    video_input: VideoInput,
    cfg: Config,
) -> List[SampledFrame]:
    """
    Extract frames at regular intervals, respecting the trim range.
    Uses a trim-aware on-disk cache (.trim_<start>_<end> marker) so re-runs
    skip re-decoding when the trim is unchanged.
    """
    video_path = video_input.path
    video_name = Path(video_path).stem
    # Own cache dir (not "frames/") so this pipeline never clobbers the touchdown
    # pipeline's sampled frames, which use a different sample interval.
    frames_dir = Path(cfg.data_dir) / "frames_yards" / video_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[frames] Cannot open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = cfg.sample_interval
    resize = cfg.frame_resize

    start_frame = video_input.start_frame or 0
    end_frame = video_input.end_frame or total

    cache_marker = frames_dir / f".trim_{start_frame}_{end_frame}_int{interval}"
    existing = sorted(frames_dir.glob("frame_*.jpg"))
    if existing and cache_marker.exists():
        samples = []
        for si, p in enumerate(existing):
            idx = int(p.stem.split("_")[1])
            samples.append(SampledFrame(
                video_name=video_name,
                video_path=video_path,
                sample_index=si,
                frame_index=idx,
                timestamp_sec=round(idx / fps, 3),
                image_path=str(p),
            ))
        cap.release()
        print(f"[frames] {video_name}: {len(samples)} cached frames "
              f"(trim {start_frame}–{end_frame}, interval {interval})")
        return samples

    # Clear stale frames if re-extracting with a different trim/interval
    for f in existing:
        f.unlink()
    for m in frames_dir.glob(".trim_*"):
        m.unlink()

    sample_count = max(0, (end_frame - start_frame) // interval)
    print(f"[frames] {video_name}: sampling every {interval} frames "
          f"from {start_frame}–{end_frame} "
          f"(~{sample_count} samples, {start_frame/fps:.1f}s–{end_frame/fps:.1f}s)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    samples = []
    frame_idx = start_frame
    si = 0
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
                sample_index=si,
                frame_index=frame_idx,
                timestamp_sec=round(frame_idx / fps, 3),
                image_path=str(path),
            ))
            si += 1
        frame_idx += 1
    cap.release()

    cache_marker.touch()
    print(f"[frames] {video_name}: {len(samples)} frames sampled")
    return samples


def sample_all_frames(video_inputs: List[VideoInput], cfg: Config) -> List[SampledFrame]:
    """Sample frames from all videos."""
    all_samples = []
    for vi in video_inputs:
        all_samples.extend(sample_frames_from_video(vi, cfg))
    print(f"[frames] Total: {len(all_samples)} sampled frames across "
          f"{len(video_inputs)} video(s)")
    return all_samples


# ============================================================
# Phase 1 — Classical CV Yard Line Detection
# ============================================================

@dataclass
class DetectedLine:
    """
    A single detected yard line. In this footage yard lines appear as
    near-horizontal segments stacked vertically, so they are ordered top-to-bottom
    by `y_center` (the line's y-coordinate at the frame's horizontal centre) — a
    stable, bounded position key for near-horizontal lines.
    """
    y_center: float                 # y-coordinate where the line crosses x = width/2
    theta_deg: float                # orientation in degrees from horizontal
    pt1: Tuple[int, int]            # endpoint 1 (for drawing)
    pt2: Tuple[int, int]            # endpoint 2 (for drawing)


@dataclass
class CVFrameResult:
    """Classical-CV result for one sampled frame."""
    video_name: str
    sample_index: int
    frame_index: int
    timestamp_sec: float
    lines: List[DetectedLine]
    vanishing_point: Optional[Tuple[float, float]]


def field_mask(frame: np.ndarray, cfg: Config) -> Optional[np.ndarray]:
    """
    Isolate the green playing field via HSV color-range thresholding.
    Returns a uint8 mask, or None if the frame has essentially no field
    (fewer than min_green_fraction of pixels green — e.g. menus/replays).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg.hsv_green_lower, dtype=np.uint8)
    upper = np.array(cfg.hsv_green_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Close small holes so painted lines inside the field stay "inside" it
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    green_fraction = float(np.count_nonzero(mask)) / mask.size
    if green_fraction < cfg.min_green_fraction:
        return None
    return mask


def line_pixel_mask(frame: np.ndarray, green_mask: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Within the field region, extract bright (white) pixels — candidate yard-line
    pixels — then morphologically suppress vertical structures (player outlines,
    HUD) while preserving thin near-horizontal lines.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, cfg.white_thresh, 255, cv2.THRESH_BINARY)

    # Keep only white pixels that fall inside the field
    field_dilated = cv2.dilate(
        green_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    candidate = cv2.bitwise_and(white, field_dilated)

    # Horizontal kernel: erode removes thin vertical structures, dilate recovers
    # the horizontal continuity of yard lines.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, cfg.morph_h_kernel)
    eroded = cv2.erode(candidate, h_kernel, iterations=cfg.morph_iterations)
    cleaned = cv2.dilate(eroded, h_kernel, iterations=cfg.morph_iterations)
    return cleaned


def hough_lines(mask: np.ndarray, cfg: Config) -> List[Tuple[int, int, int, int]]:
    """Run the probabilistic Hough transform; return [(x1,y1,x2,y2), ...]."""
    segments = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.hough_min_line_len,
        maxLineGap=cfg.hough_max_line_gap,
    )
    if segments is None:
        return []
    out = []
    for seg in segments:
        x1, y1, x2, y2 = seg[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angle = min(angle, 180 - angle)  # fold to [0, 90]
        if angle <= cfg.max_line_angle_deg:
            out.append((int(x1), int(y1), int(x2), int(y2)))
    return out


def _line_intersection(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> Optional[Tuple[float, float]]:
    """Intersection point of two segments treated as infinite lines, or None."""
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return px, py


def vanishing_point_ransac(
    segments: List[Tuple[int, int, int, int]], cfg: Config
) -> Tuple[Optional[Tuple[float, float]], List[Tuple[int, int, int, int]]]:
    """
    Estimate the field vanishing point by clustering pairwise line intersections,
    then keep only lines that point toward the dominant cluster (rejecting HUD,
    sideline clutter, and player edges).

    Returns (vanishing_point, inlier_segments). With too few lines to estimate a
    VP, returns (None, segments) unchanged.
    """
    if len(segments) < cfg.min_lines_for_vp:
        return None, segments

    # Collect pairwise intersections
    points: List[Tuple[float, float]] = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            pt = _line_intersection(segments[i], segments[j])
            if pt is not None and abs(pt[0]) < 1e5 and abs(pt[1]) < 1e5:
                points.append(pt)

    if not points:
        return None, segments

    # Greedy clustering: the largest cluster's centroid is the vanishing point
    pts = np.array(points)
    best_center = None
    best_count = 0
    eps2 = cfg.vp_cluster_eps ** 2
    for cx, cy in points:
        d2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
        members = pts[d2 <= eps2]
        if len(members) > best_count:
            best_count = len(members)
            best_center = members.mean(axis=0)

    if best_center is None:
        return None, segments
    vp = (float(best_center[0]), float(best_center[1]))

    # Keep lines whose direction aligns with the bearing to the VP
    inliers = []
    tol = np.radians(cfg.vp_inlier_tol_deg)
    for seg in segments:
        x1, y1, x2, y2 = seg
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        line_ang = np.arctan2(y2 - y1, x2 - x1)
        vp_ang = np.arctan2(vp[1] - my, vp[0] - mx)
        diff = abs(((line_ang - vp_ang + np.pi) % (2 * np.pi)) - np.pi)
        diff = min(diff, np.pi - diff)  # lines are undirected
        if diff <= tol:
            inliers.append(seg)

    if len(inliers) < cfg.min_lines_for_vp:
        # Consensus too weak — trust nothing rather than emit garbage
        return vp, inliers
    return vp, inliers


def _segment_to_line(
    seg: Tuple[int, int, int, int], width: int
) -> DetectedLine:
    """
    Convert a raw segment to a DetectedLine, positioned by its y-coordinate at the
    frame's horizontal centre (x = width/2). For near-horizontal yard lines this is
    a stable, in-frame position; an x-intercept would diverge as lines flatten.
    """
    x1, y1, x2, y2 = seg
    xc = width / 2.0
    if x2 == x1:
        y_center = (y1 + y2) / 2.0
    else:
        slope = (y2 - y1) / (x2 - x1)
        y_center = y1 + slope * (xc - x1)
    theta = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    theta = min(theta, 180 - theta)  # fold to [0, 90]
    return DetectedLine(
        y_center=float(y_center),
        theta_deg=float(theta),
        pt1=(int(x1), int(y1)),
        pt2=(int(x2), int(y2)),
    )


def filter_by_spacing(
    lines: List[DetectedLine], cfg: Config
) -> List[DetectedLine]:
    """
    Order lines top-to-bottom and keep the largest run with consistent spacing.

    Yard lines are evenly spaced on the field; under perspective the vertical gaps
    change smoothly, so successive gap *ratios* stay near 1. Lines that break the
    progression (stray HUD/sideline detections) are dropped. Near-duplicate lines
    (within line_merge_px) are merged first.
    """
    if not lines:
        return []

    lines = sorted(lines, key=lambda l: l.y_center)

    # Merge near-duplicates
    merged: List[DetectedLine] = [lines[0]]
    for l in lines[1:]:
        if abs(l.y_center - merged[-1].y_center) <= cfg.line_merge_px:
            # keep the more-horizontal of the two (lower theta) as representative
            if l.theta_deg < merged[-1].theta_deg:
                merged[-1] = l
        else:
            merged.append(l)

    if len(merged) <= 2:
        return merged

    gaps = [merged[i + 1].y_center - merged[i].y_center for i in range(len(merged) - 1)]

    # Greedily grow the longest consistent run by gap-ratio similarity
    best_run: List[int] = [0, 1]
    cur_run: List[int] = [0, 1]
    for i in range(1, len(gaps)):
        prev_gap = gaps[i - 1]
        gap = gaps[i]
        ratio = gap / prev_gap if prev_gap > 1e-6 else float("inf")
        if abs(ratio - 1.0) <= cfg.spacing_tolerance:
            cur_run.append(i + 1)
        else:
            if len(cur_run) > len(best_run):
                best_run = cur_run
            cur_run = [i, i + 1]
    if len(cur_run) > len(best_run):
        best_run = cur_run

    return [merged[i] for i in best_run]


def detect_yard_lines(
    frame: np.ndarray,
    sample: SampledFrame,
    cfg: Config,
) -> CVFrameResult:
    """
    Full Phase-1 pipeline for one frame. Never raises — on any failure or
    field-less frame it returns an empty `lines` list.
    """
    empty = CVFrameResult(
        video_name=sample.video_name,
        sample_index=sample.sample_index,
        frame_index=sample.frame_index,
        timestamp_sec=sample.timestamp_sec,
        lines=[],
        vanishing_point=None,
    )
    try:
        gmask = field_mask(frame, cfg)
        if gmask is None:
            return empty  # no field visible (menu/replay/etc.)

        lmask = line_pixel_mask(frame, gmask, cfg)
        segments = hough_lines(lmask, cfg)
        if not segments:
            return empty

        vp, inliers = vanishing_point_ransac(segments, cfg)
        if not inliers:
            return empty

        width = frame.shape[1]
        lines = [_segment_to_line(s, width) for s in inliers]
        lines = filter_by_spacing(lines, cfg)

        return CVFrameResult(
            video_name=sample.video_name,
            sample_index=sample.sample_index,
            frame_index=sample.frame_index,
            timestamp_sec=sample.timestamp_sec,
            lines=lines,
            vanishing_point=vp,
        )
    except Exception as e:  # robustness: a bad frame must not kill the run
        print(f"[cv] frame {sample.frame_index}: error ({e}), skipping")
        return empty


def detect_yard_lines_all(
    samples: List[SampledFrame], cfg: Config
) -> List[CVFrameResult]:
    """Run Phase-1 classical CV across all sampled frames."""
    results: List[CVFrameResult] = []
    total = len(samples)
    no_field = 0
    t0 = time.time()
    for i, sample in enumerate(samples):
        frame = cv2.imread(sample.image_path)
        if frame is None:
            results.append(detect_yard_lines(np.zeros((1, 1, 3), np.uint8), sample, cfg))
            continue
        res = detect_yard_lines(frame, sample, cfg)
        if not res.lines:
            no_field += 1
        results.append(res)
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [cv {i+1}/{total}] lines this frame: {len(res.lines)}")
    elapsed = time.time() - t0
    with_lines = sum(1 for r in results if r.lines)
    print(f"[cv] Done: {total} frames in {elapsed:.1f}s "
          f"({with_lines} with lines, {no_field} without)")
    return results


# ============================================================
# Phase 2 — VLM Yard Number Reading
# ============================================================

def _is_smolvlm(model_id: str) -> bool:
    """Check if model_id is a SmolVLM variant."""
    return "SmolVLM" in model_id or "smolvlm" in model_id.lower()


def load_vlm(cfg: Config):
    """
    Load the VLM with device-appropriate settings. Supports both Qwen2.5-VL and
    SmolVLM2 model families. Returns (model, processor).
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    dtype = getattr(torch, cfg.torch_dtype)
    print(f"[vlm] Loading {cfg.model_id} on {cfg.device} ({cfg.torch_dtype}) ...")

    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[vlm] VRAM: {mem:.1f} GB")

    load_kwargs = dict(torch_dtype=dtype)

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
        load_kwargs["_attn_implementation"] = "eager"
        load_kwargs["device_map"] = None

    if _is_smolvlm(cfg.model_id):
        model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, **load_kwargs)
    else:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cfg.model_id, **load_kwargs
            )
        except (ImportError, Exception):
            model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, **load_kwargs)

    if cfg.device == "mps":
        model = model.to("mps")
    model.eval()

    proc_kwargs = {}
    if not _is_smolvlm(cfg.model_id):
        proc_kwargs["min_pixels"] = cfg.min_pixels
        proc_kwargs["max_pixels"] = cfg.max_pixels
    processor = AutoProcessor.from_pretrained(cfg.model_id, **proc_kwargs)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[vlm] Loaded: {param_count:.0f}M parameters on {cfg.device}")
    return model, processor


@dataclass
class VLMReading:
    """Yard numbers read by the VLM for one anchor frame."""
    video_name: str
    sample_index: int
    frame_index: int
    timestamp_sec: float
    yard_numbers: List[int]
    raw_response: str


def _parse_yard_response(text: str) -> List[int]:
    """
    Parse a VLM response into an ordered list of yard numbers.
    Accepts 'YARD_LINES: 20, 25, 30' and tolerates extra prose. 'NONE' or
    unparseable output -> []. Yard values are clamped to the valid 0–50 range
    and de-duplicated while preserving left-to-right order.
    """
    m = re.search(r"YARD_LINES?\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    payload = m.group(1) if m else text
    if "NONE" in payload.upper():
        return []

    nums = re.findall(r"\d+", payload)
    out: List[int] = []
    for n in nums:
        v = int(n)
        if 0 <= v <= 50 and v not in out:
            out.append(v)
    return out


def _save_readings_cache(readings: List[VLMReading], path: Path) -> None:
    """Persist VLM readings to JSON for fault tolerance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in readings], f, indent=2)


def run_vlm_readings(
    samples: List[SampledFrame],
    model,
    processor,
    cfg: Config,
) -> List[VLMReading]:
    """
    Run the VLM on every vlm_interval-th sampled frame to read painted yard
    numbers. Results are cached to <output_dir>/vlm_readings_cache.json; the
    cache is reused whenever its length matches the expected anchor count (same
    length-keyed caveat as the touchdown pipeline — clear it if you change the
    prompt or model but keep the same frames).
    """
    device = cfg.device
    anchors = [s for s in samples if s.sample_index % cfg.vlm_interval == 0]

    cache_path = Path(cfg.output_dir) / "vlm_readings_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if len(cached) == len(anchors):
            print(f"[vlm] Loaded {len(cached)} cached readings from {cache_path}")
            return [VLMReading(**c) for c in cached]
        print(f"[vlm] Cache mismatch ({len(cached)} vs {len(anchors)}), re-running.")

    total = len(anchors)
    print(f"[vlm] Reading yard numbers on {total} anchor frames "
          f"(every {cfg.vlm_interval} samples) ...")
    smolvlm = _is_smolvlm(cfg.model_id)
    readings: List[VLMReading] = []
    t0 = time.time()

    for i, sample in enumerate(anchors):
        image = Image.open(sample.image_path).convert("RGB")

        if smolvlm:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": cfg.detection_prompt},
                ],
            }]
        else:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": cfg.detection_prompt},
                ],
            }]

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input], images=[image], padding=True, return_tensors="pt"
        )
        if device == "cuda":
            inputs = inputs.to("cuda")
        elif device == "mps":
            inputs = inputs.to("mps")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False
            )
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        yard_numbers = _parse_yard_response(response)
        readings.append(VLMReading(
            video_name=sample.video_name,
            sample_index=sample.sample_index,
            frame_index=sample.frame_index,
            timestamp_sec=sample.timestamp_sec,
            yard_numbers=yard_numbers,
            raw_response=response.strip(),
        ))

        if (i + 1) % 5 == 0 or (i + 1) == total:
            rate = (i + 1) / (time.time() - t0)
            print(f"  [vlm {i+1}/{total}] {rate:.2f} frames/sec | "
                  f"last yards: {yard_numbers}")
            _save_readings_cache(readings, cache_path)

        del inputs, output_ids, generated_ids
        if device == "cuda":
            torch.cuda.empty_cache()

    _save_readings_cache(readings, cache_path)
    elapsed = time.time() - t0
    print(f"[vlm] Done: {total} anchor frames in {elapsed:.0f}s")
    return readings


# ============================================================
# Phase 3 — Fusion (CV pixels + VLM yard numbers)
# ============================================================

@dataclass
class FrameYardAssignment:
    """Fused per-frame result: pixel positions tagged with yard values."""
    video_name: str
    sample_index: int
    frame_index: int
    timestamp_sec: float
    lines: List[Dict]              # [{pixel_y, yard_value, confidence}]
    anchor_source: str             # "vlm" | "propagated" | "none"


def _assign_yards_to_lines(
    cv_lines: List[DetectedLine],
    yard_numbers: List[int],
    base_conf: float,
) -> List[Dict]:
    """
    Assign yard values to CV lines (both ordered top-to-bottom / first-to-last).
    The VLM count is authoritative: when counts differ we match as many as possible
    and leave the rest with yard_value=None at reduced confidence.
    """
    cv_sorted = sorted(cv_lines, key=lambda l: l.y_center)
    yards = list(yard_numbers)

    if not yards:
        return [{"pixel_y": round(l.y_center, 1), "yard_value": None,
                 "confidence": round(base_conf * 0.5, 3)} for l in cv_sorted]

    out: List[Dict] = []
    if len(cv_sorted) == len(yards):
        for l, y in zip(cv_sorted, yards):
            out.append({"pixel_y": round(l.y_center, 1), "yard_value": y,
                        "confidence": round(base_conf, 3)})
    elif len(cv_sorted) > len(yards):
        # Too many CV lines: keep the best-spaced subset of size len(yards).
        keep = _best_spaced_subset(cv_sorted, len(yards))
        keep_set = {id(l) for l in keep}
        yi = 0
        for l in cv_sorted:
            if id(l) in keep_set and yi < len(yards):
                out.append({"pixel_y": round(l.y_center, 1), "yard_value": yards[yi],
                            "confidence": round(base_conf * 0.8, 3)})
                yi += 1
            else:
                out.append({"pixel_y": round(l.y_center, 1), "yard_value": None,
                            "confidence": round(base_conf * 0.5, 3)})
    else:
        # Fewer CV lines than yards: assign the leading yard subset.
        for i, l in enumerate(cv_sorted):
            out.append({"pixel_y": round(l.y_center, 1), "yard_value": yards[i],
                        "confidence": round(base_conf * 0.7, 3)})
    return out


def _best_spaced_subset(lines: List[DetectedLine], k: int) -> List[DetectedLine]:
    """Pick k lines whose successive gaps are most uniform (lowest gap variance)."""
    if k >= len(lines):
        return lines
    if k <= 0:
        return []
    n = len(lines)
    best_combo = None
    best_score = float("inf")
    # n is small (handful of lines); brute force over contiguous windows is enough
    for start in range(0, n - k + 1):
        window = lines[start:start + k]
        gaps = [window[i + 1].y_center - window[i].y_center for i in range(k - 1)]
        if not gaps:
            return window
        var = float(np.var(gaps))
        if var < best_score:
            best_score = var
            best_combo = window
    return best_combo if best_combo is not None else lines[:k]


def fuse_lines_with_yards(
    cv_results: List[CVFrameResult],
    vlm_readings: List[VLMReading],
    cfg: Config,
) -> List[FrameYardAssignment]:
    """
    Fuse CV pixel positions with VLM yard numbers.

    For each frame we find the nearest VLM anchor (by sample index, per video).
    Frames at an anchor are tagged "vlm"; intermediate frames "propagated".
    With no VLM readings at all, lines are emitted with yard_value=None.
    """
    # Index readings per video for nearest-anchor lookup
    by_video_readings: Dict[str, List[VLMReading]] = {}
    for r in vlm_readings:
        by_video_readings.setdefault(r.video_name, []).append(r)
    for rs in by_video_readings.values():
        rs.sort(key=lambda r: r.sample_index)

    anchor_idx_set = {(r.video_name, r.sample_index) for r in vlm_readings}
    assignments: List[FrameYardAssignment] = []

    for cv in cv_results:
        readings = by_video_readings.get(cv.video_name, [])

        if not cv.lines:
            assignments.append(FrameYardAssignment(
                video_name=cv.video_name, sample_index=cv.sample_index,
                frame_index=cv.frame_index, timestamp_sec=cv.timestamp_sec,
                lines=[], anchor_source="none",
            ))
            continue

        if not readings:
            lines = _assign_yards_to_lines(cv.lines, [], base_conf=0.4)
            assignments.append(FrameYardAssignment(
                video_name=cv.video_name, sample_index=cv.sample_index,
                frame_index=cv.frame_index, timestamp_sec=cv.timestamp_sec,
                lines=lines, anchor_source="none",
            ))
            continue

        # Nearest VLM reading by sample-index distance
        nearest = min(readings, key=lambda r: abs(r.sample_index - cv.sample_index))
        is_anchor = (cv.video_name, cv.sample_index) in anchor_idx_set
        base_conf = 0.9 if is_anchor else 0.6
        source = "vlm" if is_anchor else "propagated"

        lines = _assign_yards_to_lines(cv.lines, nearest.yard_numbers, base_conf)
        assignments.append(FrameYardAssignment(
            video_name=cv.video_name, sample_index=cv.sample_index,
            frame_index=cv.frame_index, timestamp_sec=cv.timestamp_sec,
            lines=lines, anchor_source=source,
        ))

    tagged = sum(1 for a in assignments if any(l["yard_value"] is not None for l in a.lines))
    print(f"[fuse] {len(assignments)} frames fused "
          f"({tagged} with at least one yard-tagged line)")
    return assignments


# ============================================================
# Output — per-frame JSON, events JSON/CSV, annotated frames, timeline
# ============================================================

def _video_metadata(video_inputs: List[VideoInput]) -> List[dict]:
    """Collect basic metadata for each input video."""
    meta = []
    for vi in video_inputs:
        cap = cv2.VideoCapture(vi.path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        sf = vi.start_frame or 0
        ef = vi.end_frame or total
        meta.append({
            "filename": Path(vi.path).name,
            "source_path": vi.path,
            "fps": round(fps, 2),
            "total_frames": total,
            "duration_seconds": round(total / fps, 2) if fps else 0,
            "trim_start_frame": sf,
            "trim_end_frame": ef,
        })
    return meta


def _build_events(assignments: List[FrameYardAssignment]) -> List[dict]:
    """
    Collapse consecutive frames sharing the same visible yard range (min–max of
    tagged yard values) into events.
    """
    events: List[dict] = []
    event_counter = 0

    by_video: Dict[str, List[FrameYardAssignment]] = {}
    for a in assignments:
        by_video.setdefault(a.video_name, []).append(a)

    for video_name, frames in by_video.items():
        frames.sort(key=lambda a: a.frame_index)
        run: List[FrameYardAssignment] = []
        run_range: Optional[Tuple[int, int]] = None

        def flush(run, run_range):
            nonlocal event_counter
            if not run or run_range is None:
                return
            event_counter += 1
            events.append({
                "event_id": event_counter,
                "video_name": video_name,
                "start_frame": run[0].frame_index,
                "end_frame": run[-1].frame_index,
                "start_timestamp_sec": run[0].timestamp_sec,
                "end_timestamp_sec": run[-1].timestamp_sec,
                "yard_min": run_range[0],
                "yard_max": run_range[1],
                "num_frames": len(run),
            })

        for a in frames:
            yards = [l["yard_value"] for l in a.lines if l["yard_value"] is not None]
            if not yards:
                flush(run, run_range)
                run, run_range = [], None
                continue
            yr = (min(yards), max(yards))
            if run_range is None or yr == run_range:
                run.append(a)
                run_range = yr
            else:
                flush(run, run_range)
                run, run_range = [a], yr
        flush(run, run_range)

    return events


def _annotate_frames(
    assignments: List[FrameYardAssignment],
    cv_results: List[CVFrameResult],
    samples: List[SampledFrame],
    cfg: Config,
) -> int:
    """
    Draw detected lines (red) with yard-number labels onto each sampled frame and
    save as JPEGs under <output_dir>/annotated/. Returns the count written.
    """
    out_dir = Path(cfg.annotated_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index CV geometry and sample image paths by (video, frame)
    cv_by_key = {(c.video_name, c.frame_index): c for c in cv_results}
    img_by_key = {(s.video_name, s.frame_index): s.image_path for s in samples}

    written = 0
    for a in assignments:
        key = (a.video_name, a.frame_index)
        img_path = img_by_key.get(key)
        if img_path is None:
            continue
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        cv = cv_by_key.get(key)
        geom_by_y = {}
        if cv:
            for dl in cv.lines:
                geom_by_y[round(dl.y_center, 1)] = dl

        for ln in a.lines:
            py = ln["pixel_y"]
            dl = geom_by_y.get(round(py, 1))
            if dl is not None:
                cv2.line(frame, dl.pt1, dl.pt2, (0, 0, 255), 2)
                # label at the left end of the segment
                tx, ty = dl.pt1 if dl.pt1[0] < dl.pt2[0] else dl.pt2
            else:
                y = int(py)
                cv2.line(frame, (0, y), (frame.shape[1], y), (0, 0, 255), 2)
                tx, ty = 5, y
            label = str(ln["yard_value"]) if ln["yard_value"] is not None else "?"
            cv2.putText(frame, label, (max(0, tx - 4), max(15, ty - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out_path = out_dir / f"{a.video_name}_frame_{a.frame_index:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        written += 1

    print(f"[output] Annotated {written} frame(s) -> {out_dir}/")
    return written


def _plot_timeline(assignments: List[FrameYardAssignment], cfg: Config) -> None:
    """Plot the visible yard range (min–max tagged yard value) over time per video."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[output] matplotlib not available, skipping timeline plot.")
        return

    by_video: Dict[str, List[FrameYardAssignment]] = {}
    for a in assignments:
        by_video.setdefault(a.video_name, []).append(a)

    n = len(by_video)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), squeeze=False)

    for idx, (vname, frames) in enumerate(by_video.items()):
        ax = axes[idx, 0]
        frames.sort(key=lambda a: a.timestamp_sec)
        ts, ymin, ymax = [], [], []
        for a in frames:
            yards = [l["yard_value"] for l in a.lines if l["yard_value"] is not None]
            if yards:
                ts.append(a.timestamp_sec)
                ymin.append(min(yards))
                ymax.append(max(yards))
        if ts:
            ax.fill_between(ts, ymin, ymax, alpha=0.3, color="green",
                            label="visible yard range")
            ax.plot(ts, ymin, ".", color="darkgreen", markersize=3)
            ax.plot(ts, ymax, ".", color="darkgreen", markersize=3)
        ax.set_title(f"{vname} — visible yard lines over time")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Yard line value")
        ax.set_ylim(-2, 52)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(Path(cfg.timeline_path), dpi=150)
    plt.close()
    print(f"[output] Timeline -> {cfg.timeline_path}")


def build_output(
    assignments: List[FrameYardAssignment],
    cv_results: List[CVFrameResult],
    samples: List[SampledFrame],
    video_inputs: List[VideoInput],
    cfg: Config,
) -> dict:
    """Write all output artifacts and return the per-frame output dict."""
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    events = _build_events(assignments)

    output = {
        "pipeline": {
            "method": "Hybrid classical-CV line detection + VLM yard-number reading",
            "model": cfg.model_id,
            "device": cfg.device,
            "sample_interval": cfg.sample_interval,
            "vlm_interval": cfg.vlm_interval,
        },
        "video_metadata": _video_metadata(video_inputs),
        "summary": {
            "total_frames": len(assignments),
            "frames_with_lines": sum(1 for a in assignments if a.lines),
            "total_events": len(events),
        },
        "frames": [
            {
                "video_name": a.video_name,
                "frame_index": a.frame_index,
                "timestamp_sec": a.timestamp_sec,
                "anchor_source": a.anchor_source,
                "lines": a.lines,
            }
            for a in assignments
        ],
    }

    # Per-frame JSON
    pf_path = Path(cfg.per_frame_json)
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pf_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[output] Per-frame JSON -> {pf_path}")

    # Events JSON
    with open(cfg.events_json, "w") as f:
        json.dump(events, f, indent=2)
    print(f"[output] Events JSON -> {cfg.events_json}")

    # Events CSV
    if events:
        fieldnames = list(events[0].keys())
        with open(cfg.events_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)
        print(f"[output] Events CSV -> {cfg.events_csv}")

    # Annotated frames + timeline
    _annotate_frames(assignments, cv_results, samples, cfg)
    _plot_timeline(assignments, cfg)

    return output


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(cfg: Config) -> Optional[dict]:
    """Execute the full yard-line detection pipeline."""
    print("=" * 60)
    print("  Yard Line Detection — Hybrid CV + VLM Pipeline")
    print(f"  Model:  {cfg.model_id}")
    print(f"  Device: {cfg.device} ({cfg.torch_dtype})")
    print("=" * 60)

    # Stage 1: Acquire videos
    print("\n" + "=" * 60 + "\n  STAGE 1: Video Acquisition\n" + "=" * 60)
    video_inputs = acquire_videos(cfg)

    # Stage 2: Sample frames
    print("\n" + "=" * 60 + "\n  STAGE 2: Sample Frames\n" + "=" * 60)
    samples = sample_all_frames(video_inputs, cfg)
    if not samples:
        print("ERROR: No frames sampled.")
        return None

    # Stage 3: Phase 1 — classical CV line detection
    print("\n" + "=" * 60 + "\n  STAGE 3: Phase 1 — Classical CV Line Detection\n" + "=" * 60)
    cv_results = detect_yard_lines_all(samples, cfg)

    # Stage 4: Phase 2 — VLM yard-number reading
    print("\n" + "=" * 60 + "\n  STAGE 4: Phase 2 — VLM Yard-Number Reading\n" + "=" * 60)
    model, processor = load_vlm(cfg)
    vlm_readings = run_vlm_readings(samples, model, processor, cfg)
    del model, processor
    if cfg.device == "cuda":
        torch.cuda.empty_cache()

    # Stage 5: Phase 3 — fusion
    print("\n" + "=" * 60 + "\n  STAGE 5: Phase 3 — Fusion\n" + "=" * 60)
    assignments = fuse_lines_with_yards(cv_results, vlm_readings, cfg)

    # Stage 6: Output
    print("\n" + "=" * 60 + "\n  STAGE 6: Output\n" + "=" * 60)
    output = build_output(assignments, cv_results, samples, video_inputs, cfg)

    # Summary
    print("\n" + "=" * 60 + "\n  PIPELINE COMPLETE\n" + "=" * 60)
    print(f"  Videos:        {len(video_inputs)}")
    print(f"  Frames:        {len(samples)}")
    print(f"  VLM anchors:   {len(vlm_readings)}")
    print(f"  Yard events:   {output['summary']['total_events']}")
    print(f"\n  Outputs:")
    print(f"    Per-frame JSON: {cfg.per_frame_json}")
    print(f"    Events JSON:    {cfg.events_json}")
    print(f"    Events CSV:     {cfg.events_csv}")
    print(f"    Annotated:      {cfg.annotated_dir}/")
    print(f"    Timeline:       {cfg.timeline_path}")
    return output


# ============================================================
# CLI & Entry Point
# ============================================================

def parse_args() -> Config:
    """Parse CLI arguments and build a Config."""
    parser = argparse.ArgumentParser(
        description="Detect yard lines in football gameplay video (CV + VLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python yard_line_detection.py --input game.mp4
  python yard_line_detection.py --input clip1.mov clip2.mp4

  Trim by seconds (use 's' suffix) or by frame range:
  python yard_line_detection.py --input game.mp4[0s:180s]
  python yard_line_detection.py --input game.mp4[0:5400]

  Model / cadence overrides:
  python yard_line_detection.py --input game.mp4 --model 3b
  python yard_line_detection.py --input game.mp4 --sample-interval 3 --vlm-interval 20
        """,
    )
    parser.add_argument(
        "--input", "-i", nargs="+", default=[],
        help="Local video file(s): .mp4, .mov, .avi, .mkv, .webm, .m4v",
    )
    parser.add_argument(
        "--model", "-m",
        choices=["auto", "7b", "3b", "2b", "500m", "256m"],
        default="auto",
        help="VLM size: 7b/3b (Qwen, GPU), 2b/500m/256m (SmolVLM2, CPU-friendly). "
             "Default: auto-detect from hardware.",
    )
    parser.add_argument(
        "--sample-interval", type=int, default=None,
        help="Run classical CV on every N-th frame (default: 5)",
    )
    parser.add_argument(
        "--vlm-interval", type=int, default=None,
        help="Run the VLM every N-th sampled frame (default: 12)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: outputs_yards/)",
    )

    args = parser.parse_args()

    if not args.input:
        print("ERROR: no --input provided. Pass at least one video file.")
        sys.exit(1)

    device, auto_model, auto_dtype = detect_device()

    model_map = {
        "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "500m": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "256m": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    }
    if args.model != "auto":
        model_id = model_map[args.model]
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
        model_id=model_id,
        torch_dtype=dtype,
        device=device,
    )

    if args.sample_interval is not None:
        cfg.sample_interval = args.sample_interval
    if args.vlm_interval is not None:
        cfg.vlm_interval = args.vlm_interval
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    # Resolve output paths against output_dir
    cfg.per_frame_json = f"{cfg.output_dir}/per_frame_yard_lines.json"
    cfg.events_json = f"{cfg.output_dir}/yard_events.json"
    cfg.events_csv = f"{cfg.output_dir}/yard_events.csv"
    cfg.annotated_dir = f"{cfg.output_dir}/annotated"
    cfg.timeline_path = f"{cfg.output_dir}/yard_timeline.png"

    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    run_pipeline(cfg)
