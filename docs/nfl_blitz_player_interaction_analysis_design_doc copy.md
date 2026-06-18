# Design Doc: NFL Blitz — Player Interaction & Distance Analysis

**Author:** Akshay  
**Date:** June 2026   
**Team:** Gameplay Analyzer
 
---

## 1. Problem Statement

Given broadcast or sideline video of an NFL game, detect and track the **ball carrier** and all other on-field players, classify each as **teammate** or **opponent**, and continuously compute the **Euclidean distance** (in real-world yards) between the ball carrier and every other player. The output enables downstream analysis of separation metrics, closing speed, tackle probability, and play-level interaction graphs.

---

## 2. Core Sub-Problems

The task decomposes into five sequential stages, each with its own failure modes:

**Stage A — Player Detection.** Locate every player bounding box (or instance mask) in each frame.

**Stage B — Ball Carrier Identification.** Determine which detected player currently possesses the ball.

**Stage C — Team Assignment.** Classify each player as belonging to Team A, Team B, or neither (officials, sideline personnel).

**Stage D — Homography & Coordinate Mapping.** Transform pixel-space positions into a calibrated field coordinate system (yards) so distances are physically meaningful.

**Stage E — Distance Computation & Tracking.** Maintain player identities across frames and compute pairwise distances from the ball carrier to all others at every timestep.

---

## 3. Implementation Approaches

### Classical Detection + Jersey Color Clustering (Baseline)

**Overview:** Use an off-the-shelf object detector, segment jersey color with HSV clustering, and estimate distances via a four-point homography from visible field markings.

**Pipeline:**

- **Detection:** YOLOv8x or RT-DETR fine-tuned on a sports-person class. Use a high-resolution input (1280×720 minimum) to catch distant players.
- **Ball Carrier ID:** Heuristic — the player nearest to the detected football whose motion vector aligns with the play direction. Football detection is a known hard problem (small, fast, occluded); fallback to optical-flow divergence around the runner.
- **Team Assignment:** Extract the dominant color from each player's torso crop (upper 40% of bounding box). Run K-Means (K=3: team A, team B, officials) in HSV space per-frame, then propagate cluster labels with a majority vote across a tracking window.
- **Homography:** Detect yard-line markings and hash marks via Hough lines or a learned line-segment detector (LETR / DeepLSD). Compute a homography H mapping four or more line intersections to their known field coordinates. Recompute H every N frames to handle camera pan/zoom.
- **Distance:** Project each player's foot-point (bottom-center of bbox) through H into field coordinates. Euclidean distance in the transformed space gives yards.

**Pros:** Simple, interpretable, each module is independently testable, runs in near real-time on a single GPU.

**Cons:** Jersey color clustering breaks on white-vs-white or similar-hue matchups, homography degrades on tight zooms with few visible lines, football detection is unreliable.

---

## 4. Recommended Architecture

For a balance of accuracy, speed, and engineering complexity, the recommended system combines elements of approaches 3.1, 3.2, and 3.4:

```
Broadcast Frame
      │
      ▼
┌─────────────┐     ┌────────────────┐
│ YOLOv8x Det │────▶│ BoT-SORT Track │
└─────────────┘     └───────┬────────┘
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
        ┌──────────┐ ┌───────────┐ ┌──────────────┐
        │ Pose Est │ │ Jersey OCR│ │ Color Cluster│
        │ (RTMPose)│ │ (CRNN)   │ │ (HSV K-Means)│
        └────┬─────┘ └─────┬─────┘ └──────┬───────┘
             │              │              │
             ▼              ▼              ▼
     ┌──────────────┐  ┌─────────┐  ┌───────────┐
     │ Ball Carrier │  │ Team ID │  │ Team ID   │
     │ Classifier   │  │ (OCR)   │  │ (fallback)│
     └──────┬───────┘  └────┬────┘  └─────┬─────┘
            │               │              │
            └───────────┬───┴──────────────┘
                        ▼
              ┌──────────────────┐
              │ Homography (yard │
              │ line keypoints)  │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ Distance Matrix  │
              │ (ball carrier ↔  │
              │  all players)    │
              └──────────────────┘
```

---

## 5. Key Technical Decisions

**Ground-contact point estimation.** Bounding box bottom-center is a 0.5–1.0 yard error source on perspective-heavy shots. Ankle keypoints from pose estimation reduce this to ~0.2 yards. For BEV approaches, the mask centroid projected onto the ground plane is most reliable.

**Handling camera motion.** Broadcast cameras pan, zoom, and cut. The homography must be recomputed continuously. On camera cuts (detected via frame-difference spike), all track IDs must be re-initialized. A pre-built field template with yard lines at known intervals (5.33 yards between lines, 53.33 yard field width) constrains the homography.

**Occlusion and pile-ups.** Near the line of scrimmage and during tackles, players overlap heavily. The tracker must handle partial occlusion (BoT-SORT's camera-motion compensation helps). Distance measurements during pile-ups are inherently noisy; flag them as low-confidence.

**Real-time vs. offline.** Detection + tracking + color clustering runs at ~25 FPS on an A100. Adding pose estimation drops to ~12 FPS. Jersey OCR on every player every frame is unnecessary — run it on high-confidence crops every 30 frames and cache the result.

---

