# MGI-Gameplay-Analysis
Gameplay Analytics - Event Detection in Gridiron games (Touchdown, Tackle etc)
# Technical Description — VLM-Based Gameplay Event Detection

## 1. Overview

This project detects touchdown events in 3D gridiron football gameplay video (NFL Blitz) using a zero-shot vision-language model (VLM). No training data or manual annotation is required — the VLM classifies each sampled frame directly.

**Source game**: NFL Blitz (Midway, 1997 / EA, 2012) — arcade-style American football.
**Target event**: Touchdown.
**Approach**: Zero-shot frame classification with Qwen2.5-VL-3B, an open-source VLM.
**Platform**: Google Colab / Kaggle with T4 GPU (16GB VRAM).

---

## 2. Why VLM Instead of Training a Classifier?

The original plan was to train an MLP classifier on backbone features extracted from manually annotated frames. This failed for two reasons:

1. **Insufficient training data.** Only 2 out of 5 YouTube videos were downloadable, yielding just 3 touchdown events — far too few to train even a simple binary classifier.

2. **OCR-based auto-annotation was too noisy.** pytesseract on stylized arcade game fonts produced mostly false positives (detecting "TD" or "SCORE" in title screens, menus, and HUD elements). The signal-to-noise ratio made semi-automated annotation impractical.

A VLM solves both problems: it requires no training data (zero-shot), and it understands visual context far better than OCR — it can distinguish a touchdown celebration from a title screen because it reasons about the entire frame, not just extracted text.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    VIDEO ACQUISITION                      │
│                                                          │
│  YouTube (yt-dlp) ──► 2 NFL Blitz clips (10 min each)   │
│  Fair use: non-commercial academic prototype             │
└──────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                    FRAME SAMPLING                         │
│                                                          │
│  30fps video ──► sample every 15 frames ──► ~2 fps       │
│  Resize to 640×480 ──► save as JPEG                      │
│  ~800 frames per 10-min clip                             │
└──────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│               VLM ZERO-SHOT INFERENCE                     │
│                                                          │
│  For each frame:                                         │
│    Qwen2.5-VL-3B receives image + structured prompt:     │
│    "Is this frame showing a TOUCHDOWN event?"            │
│    ──► "TOUCHDOWN: YES/NO" + "CONFIDENCE: 0.0–1.0"      │
│                                                          │
│  ~1.5 sec/frame on T4 GPU                                │
│  No training, no fine-tuning, no annotation              │
└──────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                   POST-PROCESSING                         │
│                                                          │
│  Per-frame predictions ──► confidence threshold (0.5)    │
│                        ──► merge contiguous positives    │
│                        ──► filter short events (≥2)      │
│                        ──► events_output.json / .csv     │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Pipeline Stages

### Stage 1 — Video Acquisition
- Download NFL Blitz gameplay footage from YouTube using `yt-dlp`
- Trim to 10 minutes per clip using `ffmpeg`
- Fair use (17 U.S.C. § 107) for non-commercial educational purposes

### Stage 2 — Frame Sampling
- Sample every 15th frame (~2 fps effective rate)
- Resize to 640×480 for consistent VLM input
- ~800 frames per 10-min clip, ~1600 total across 2 videos

### Stage 3 — VLM Inference
- Load Qwen2.5-VL-3B-Instruct (open-source, Apache 2.0 license)
- float16 precision, fits within T4's 16GB VRAM
- Each frame receives a structured prompt asking for touchdown detection
- Model returns "TOUCHDOWN: YES/NO" and "CONFIDENCE: 0.0–1.0"
- Greedy decoding (temperature=0) for reproducibility
- Predictions cached to disk every 50 frames for fault tolerance
- Total inference time: ~20–40 minutes for 1600 frames on T4

### Stage 4 — Post-Processing
- Filter frames below confidence threshold (0.5)
- Group contiguous positive frames per video
- Merge groups separated by ≤4 sample intervals
- Filter events shorter than 2 consecutive positive samples
- Each surviving group becomes one touchdown event

### Stage 5 — Output
- `events_output.json`: structured event data with video metadata, pipeline config, and per-event details
- `events_output.csv`: tabular event data
- `detection_timeline.png`: confidence plot over time with event regions highlighted

---

## 5. Output Schema

```json
{
  "pipeline": {
    "method": "VLM zero-shot inference (no training)",
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "sample_interval": 15,
    "confidence_threshold": 0.5
  },
  "video_metadata": [
    {
      "filename": "blitz_n64_longplay.mp4",
      "fps": 30,
      "total_frames": 18000,
      "duration_seconds": 600.0
    }
  ],
  "summary": {
    "total_frames_sampled": 1600,
    "touchdown_frames": 45,
    "touchdown_frame_rate": 0.028,
    "total_events_detected": 8
  },
  "events": [
    {
      "event_id": 1,
      "video_name": "blitz_n64_longplay",
      "start_frame": 2700,
      "end_frame": 2790,
      "start_timestamp_sec": 90.0,
      "end_timestamp_sec": 93.0,
      "duration_sec": 3.0,
      "avg_confidence": 0.85,
      "peak_confidence": 0.95,
      "num_positive_frames": 6,
      "detection_method": "Qwen2.5-VL-3B zero-shot"
    }
  ]
}
```

---

## 6. Model Choice: Why Qwen2.5-VL-3B?

| Requirement | Qwen2.5-VL-3B |
|---|---|
| Open-source | Yes (Apache 2.0) |
| Fits T4 GPU (16GB) | Yes, ~6GB in float16 |
| Text reading in images | Strong OCR capability |
| Visual scene understanding | Understands game UIs, celebrations, overlays |
| Instruction following | Returns structured YES/NO + confidence |
| No flash attention needed | Works with SDPA on T4 (non-Ampere) |

The 7B variant has compatibility issues on T4 GPUs. The 3B variant fits comfortably and provides sufficient accuracy for detecting visually distinct events like touchdowns in NFL (large "TOUCHDOWN" text overlays, celebration animations).

---

## 7. Limitations and Future Work

**Current limitations:**
- No ground truth evaluation (no manually annotated test set), so precision/recall cannot be computed
- ~1.5 sec/frame inference limits real-time applicability
- Sample interval of 15 frames may miss very brief events (<0.5 sec)
- VLM responses can be inconsistent across similar frames

**Possible extensions:**
- Use VLM predictions as pseudo-labels to train a lightweight CNN classifier (student-teacher distillation)
- Fine-tune Qwen2.5-VL-3B on a small set of confirmed touchdown frames
- Add temporal context by sending 2-3 consecutive frames per VLM call
- Detect additional event types (interception, fumble, field goal)

---

## 8. Dependencies

- Python 3.10+
- PyTorch 2.0+ (CUDA)
- transformers >= 4.45.0
- accelerate
- qwen-vl-utils
- yt-dlp, opencv-python-headless
- matplotlib, numpy, Pillow
- num2words
---

## 9. How to Run

### Install Dependencies
```bash
# On Google Colab or Kaggle (with T4 GPU):
!pip install -q transformers>=4.45.0 accelerate qwen-vl-utils
!pip install -q yt-dlp opencv-python-headless
```

### Usage:
```bash
    # From local video file(s):
    python gameplay_event_detection.py --input game.mp4
    python gameplay_event_detection.py --input clip1.mov clip2.mp4
    python gameplay_event_detection.py --input recordings/*.mp4

    # Download from YouTube (fair use for academic purposes):
    python gameplay_event_detection.py --download

    # Mixed — local files + downloads:
    python gameplay_event_detection.py --input game.mp4 --download

    # Force a specific model:
    python gameplay_event_detection.py --input game.mp4 --model 7b
    python gameplay_event_detection.py --input game.mp4 --model 3b

    # Adjust sampling rate (lower = more accurate, slower):
    python gameplay_event_detection.py --input game.mp4 --sample-interval 10

    # Run only on a specific time range of a video(in seconds)
    python gameplay_event_detection.py --input game.mp4[10s:50s]
```

---


