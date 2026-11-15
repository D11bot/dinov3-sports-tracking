# DINOv3 for Sports Tracking

I built this project to learn more about player tracking and to test how reliable identity matching is when detection quality changes. I started with frozen DINOv3 features and then compared appearance matching, IoU matching, and Kalman motion prediction on SoccerNet football videos.

I used the results to investigate why a tracker can look very stable with clean annotations but struggle when the bounding boxes become less accurate. I also looked at what happens when detections are missing or when extra background boxes are added.

## What I wanted to find out

1. Can DINOv3 features help me recognize the same player across different video frames?
2. How do missing detections, inaccurate boxes, and false positives affect identity matching?
3. Can a more suitable Kalman measurement noise setting reduce errors when box positions are noisy?

## How I built the experiments

I used the SoccerNet Tracking 2023 training sequences. I tested the first 120 frames of SNMOT-060 and SNMOT-061. I inspected the small objects in the images and excluded the ball using GT ID 18 for SNMOT-060 and GT ID 1 for SNMOT-061. These IDs are specific to those two sequences.

I extracted 384 dimensional features using the pretrained DINOv3 ViT-S/16 model with its weights frozen. I matched players with Hungarian assignment and compared five approaches:

| Method | What I used |
|---|---|
| IoU | Box overlap |
| DINOv3 appearance | Cosine similarity between player embeddings |
| Fixed fusion | Equal weights for normalized cosine similarity and IoU |
| Kalman motion | Predicted position and Mahalanobis distance |
| Kalman with DINOv3 | The same motion gate with an additional appearance score |

I first compared individual frames. I then built a tracker that updates identities across consecutive frames. Next, I removed selected observations to simulate detection gaps. Finally, I added missing detections, box jitter, and synthetic false positives to study tracking under less reliable input data.

## Results I observed

### 1. DINOv3 appearance matching

I compared frame 1 and frame 10 in SNMOT-060. The average cosine similarity was **0.8871** for the same identity and **0.6080** for different identities. My appearance only Top 1 retrieval found **17 of 18** identities. Hungarian matching found **18 of 18** in this frame pair.

The appearance features carried useful identity information in this example. I also found high similarity between some different players, which made me interested in combining appearance and motion information.

### 2. Recovery after missing detections

I removed selected player detections for 2, 3, 5, or 8 frames. In SNMOT-060, I tested 48 cases per method with the same input scenarios.

| Method | Successful recoveries |
|---|---:|
| IoU | 31 / 48 |
| DINOv3 appearance | 44 / 48 |
| Fixed fusion | 29 / 48 |
| Kalman motion | 48 / 48 |
| Kalman with DINOv3 | 48 / 48 |

I counted a recovery when the player received the same predicted track ID before and after the gap and the track was not assigned to another identity during the checked interval. I use this as a custom recovery metric.

Appearance matching helped more than static IoU in these tests. After adding Kalman prediction, the motion method recovered all the tested cases, and the DINOv3 features did not change the measured recovery rate in this setup.

### 3. A case where appearance matching went wrong

I investigated a detection gap for GT ID 7. At frame 44, the appearance tracker assigned its track to GT ID 10. The appearance similarity was about **0.8062**, while box IoU was **0**. The fixed fusion score was **0.4516**, below the matching threshold of **0.55**.

This helped me understand how similar football uniforms can cause identity confusion and how spatial checks can reject some incorrect matches.

### 4. Tracking with imperfect detection boxes

I tested synthetic detection errors on SNMOT-061. With clean ground truth boxes, both Kalman trackers kept the player identities stable across the first 120 frames. I then changed the detector input while keeping the original ground truth for evaluation.

| Input condition | KF Motion ID changes | KF with DINOv3 ID changes |
|---|---:|---:|
| Clean boxes | 0 | 0 |
| 20% random missing detections | 0 | 0 |
| 15% box jitter | 1,420 | 1,375 |
| One synthetic false positive per frame | 8 | 6 |

Under 20% missing detections, detection recall was about **79.46%**. With 15% box jitter, detection recall was about **79.80%** using an IoU threshold of 0.5. I used the known source identity of each synthetic detection to separately check ID continuity.

I found that noisy box locations created many more track IDs. In one run, KF Motion created **100** tracks and KF with DINOv3 created **92** tracks from **17** annotated player identities.

I also tried jitter levels of **5%, 10%, and 15%**. The number of identity changes grew quickly as the jitter increased. Adding appearance features helped in some runs but the difference was small and was not consistent across all noise levels.

### 5. Looking into the Kalman matching decisions

I recorded candidate scores, Mahalanobis distances, gate decisions, and track creation events. In the 15% jitter experiment, most new track events were linked to correct identity candidates being rejected by the motion gate.

| Cause found in candidate diagnostics | KF Motion | KF with DINOv3 |
|---|---:|---:|
| Correct candidate rejected by gate | 82 | 73 |
| Candidate passed gate but was not selected | 1 | 2 |

I noticed that the filter used a fixed measurement noise variance of **16 px²**, while the synthetic box offsets could be much larger. I decided to test different R values instead of immediately changing the appearance model.

### 6. Testing measurement noise R

I compared **R = 16, 64, 144, and 256**, using the same detection inputs and association settings. Increasing R reduced identity changes and extra track creation in the synthetic jitter experiments.

For example, on SNMOT-061 with 15% jitter:

| R variance | KF Motion ID changes | KF with DINOv3 ID changes |
|---|---:|---:|
| 16 | 1,420 | 1,375 |
| 64 | 878 | 844 |
| 144 | 345 | 359 |
| 256 | 113 | 105 |

I then used a different sequence, SNMOT-060, with a new random seed and fixed settings to check the same idea:

| Method | R = 16 | R = 256 |
|---|---:|---:|
| KF Motion ID changes | 1,221 | 196 |
| KF with DINOv3 ID changes | 1,162 | 182 |

This was the most useful part of the project for me. I found a tracking problem, followed the matching logs to a possible cause, changed one Kalman setting, and checked the result on another sequence.

The CSV files in [`results/reference/`](results/reference/) come from my original executed Kaggle Notebook. I also ran this organized Python project in Kaggle with SNMOT-061 and confirmed the main baseline, detection quality, Kalman diagnostic, and R sensitivity results again.

## How I organized the code

```text
run_experiment.py                 Main script to run my experiments
src/data.py                       SoccerNet input files and feature cache
src/features.py                   Frozen DINOv3 feature extraction
src/association.py                IoU and Hungarian matching
src/simple_tracker.py             IoU, appearance and fixed fusion trackers
src/simple_diagnostics.py         Candidate logs for the simple trackers
src/kalman.py                     Kalman position prediction and configurable R
src/kalman_tracker.py             Kalman motion and DINOv3 tracking
src/diagnostics.py                Kalman candidate and event logs
src/corruption.py                 Synthetic detection errors
src/evaluation.py                 My tracking and quality measurements
experiments/appearance_analysis.py  Frame matching experiments
experiments/plot_results.py      Plots from saved CSV data
results/reference/               Original Notebook results
examples/kaggle_run.md           How I run the project in Kaggle
tests/                           Small tests for the main components
```

## Running the project

I run the experiments with Python and PyTorch. For DINOv3, I use the pretrained model `facebook/dinov3-vits16-pretrain-lvd1689m` from Hugging Face. I put my Hugging Face access token in the `HF_TOKEN` environment variable.

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
```

With a SoccerNet sequence extracted into a folder containing `img1/` and `gt/gt.txt`:

```bash
python run_experiment.py \
  --sequence-dir data/SNMOT-061 --frames 120 --exclude-ids 1 \
  --cache cache/SNMOT-061.npz --output outputs/SNMOT-061 \
  --experiment all
```

I can also extract a selected sequence from a downloaded SoccerNet ZIP:

```bash
python run_experiment.py \
  --train-zip data/train.zip --sequence train/SNMOT-060 \
  --sequence-dir data/SNMOT-060 --frames 120 --exclude-ids 18 \
  --cache cache/SNMOT-060.npz --output outputs/SNMOT-060 \
  --experiment all
```

To run the frame matching experiment using cached features:

```bash
python -m experiments.appearance_analysis --cache cache/SNMOT-060.npz --targets 10 30 60 100
```

To run the R sensitivity experiment separately:

```bash
python run_experiment.py --sequence-dir data/SNMOT-061 \
  --exclude-ids 1 --cache cache/SNMOT-061.npz \
  --experiment calibration --jitter 0.15 --seed 42 \
  --r-values 16 64 144 256 --output outputs/r_061
```

I use `python run_experiment.py --help` to see the other options. I also put my Kaggle steps in [`examples/kaggle_run.md`](examples/kaggle_run.md).

## Experiment settings

| Setting | Value |
|---|---|
| DINOv3 model | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| Input format | SoccerNet MOT ground truth boxes, 120 frames |
| IoU threshold | 0.1 |
| Appearance similarity threshold | 0.78 |
| Fixed fusion | 0.5 normalized cosine + 0.5 IoU, threshold 0.55 |
| Kalman state | Box center x, y and velocities vx, vy |
| Initial Kalman R | diag(16, 16) |
| Mahalanobis gate | d² ≤ 9.21 |
| Kalman with DINOv3 | 0.55 motion score + 0.45 normalized cosine |
| Max track age | 10 frames in the main gap and Kalman tests |
| R sensitivity settings | 16, 64, 144, 256 px² |
| Gap settings | starts 35 and 60; lengths 2, 3, 5 and 8 |
| Random seeds | 42 for the main jitter tests; 2026 for the second sequence check |

## What I learned and what I want to test next

My main takeaway is that detection quality matters as much as the matching algorithm. Clean boxes made the tracking task easy, but localization noise exposed problems in motion gating and track management. DINOv3 helped with some identity recovery cases, while motion prediction was stronger in other situations. I learned that adding a visual model does not automatically improve every tracking decision.

The results above are from controlled experiments with ground truth boxes and synthetic detection errors. I report recovery rate, identity changes, and track contamination using the definitions in my evaluation code. I use them to compare my approaches within this project. I would like to add real person detections, more challenging video sequences, and standard MOT metrics such as HOTA and IDF1 next.

I keep the repository focused on code and small result tables. I use SoccerNet data and DINOv3 weights according to their respective access conditions. See [`LICENSE_NOTICE.md`](LICENSE_NOTICE.md) for more information.

## Plotting results

I can also plot a generated CSV directly:

```bash
python -m experiments.plot_results --csv outputs/r_061/r_sensitivity.csv \
  --x R_Variance --y Identity_Changes --output outputs/r_061/identity_changes.png
```
