# Running LEAP as in Tetris

This repository is the LEAP code Tetris compares against, with the changes and adaptations Tetris's runs used.
`README.md` is LEAP's own documentation; this file lists what Tetris ran and how.

## What is here

- LEAP's code as run: see the commit "add the code as run for the Tetris experiments".
- `vehicle_reid2/`: fast-reid at `c9bc3ce` (vendored) plus LEAP's dataset registrations, configs, and scripts.
- `env/`: the exact conda and pip environments, OS, and CUDA (`env/README.md`).
- `experiments/configs/workspace/`: the per-dataset configs the runs read.
- `experiments/configs/reviewed-regions/`: Amsterdam and CalDoT configs with reviewed `stop_area` regions.
- `experiments/run/run_leap.py`: LEAP's multi-video main loop with a per-clip export of LEAP's state.
- `experiments/run/matched_detector.py`, `experiments/prep/motion_patterns.py`, `experiments/eval/export_tracks.py`:
  the adaptations below.
- `experiments/data/`, `experiments/eval/`: dataset conversion and export scripts
  (paths inside are hard-coded to the machine they ran on; edit them before use).

Not here: videos, labels, and weights.

- `weights/yolov5x.pt`: YOLOv5 v5.0 COCO weights (links in `README.md`), the labeling detector for CalDoT and Amsterdam.
- `pretrained/veriwild_bot_R50-ibn.pth`: fast-reid's VeRi-Wild BoT R50-ibn model (`vehicle_reid2/MODEL_ZOO.md`),
  the starting point of re-identification training.
- B3D's RetinaNet model, and Tetris's fine-tuned YOLOv5 checkpoints for the matched-detector runs.

## The data workspace

LEAP reads labels from a hard-coded path (`settings/settings.py`: `/data/alenachao/data/leap-dataset/labels/`),
so the workspace is mounted at `/data/alenachao/data/leap-dataset`:

```
configs/<config>.yaml              experiments/configs/workspace/ (or reviewed-regions/)
labels/<full_name>/<clip>/{train,test}/label_<clip>.txt   pseudo labels (step 1)
logs/<full_name>/model_final.pth   re-identification model (step 3)
motion_patterns/<full_name>/       motion patterns (step 4)
models/yolov5/, models/retinanet/  detector weights
```

Clips are Tetris's test and training clips renamed to `N.mp4` (`experiments/data/rename_leap_videos.py`),
one directory per split.

| Tetris dataset | Config | `full_name` | Labels directory | Detector | Motion patterns |
| --- | --- | --- | --- | --- | --- |
| CalDoT1 | `caldot1.yaml` | `caldot1` | `caldot1-y5` | YOLOv5x | `fixed_files/preprocessed` (LEAP's default) |
| CalDoT2 | `caldot2.yaml` | `caldot2` | `caldot2-y5` | YOLOv5x | `fixed_files/preprocessed` |
| Amsterdam | `amsterdam.yaml` | `amsterdam-y5` | `amsterdam-y5` | YOLOv5x | workspace `motion_patterns/` |
| B3D1-4 | `jnc0/2/6/7.yaml` | `jnc0/2/6/7` | same | RetinaNet | workspace `motion_patterns/` |

For CalDoT, LEAP looks up `labels/<full_name>`, so link `labels/caldot1 -> caldot1-y5` (and caldot2).

## Preparation (per dataset, from the repository root; `OUT` = output root, `R` = lower-case ReID name)

```sh
# 1. Pseudo labels of the train and test clips (B3D: --use_retinanet --retinanet_path <workspace>/models/retinanet)
python video_labeling/video_tracker.py --video_dir <clips>/<split> --save_path $OUT/labels/<labels>/<split> \
    --fps 15 --yolo_weight weights/yolov5x.pt
# 2. Re-identification crops (train, query, gallery)
python vehicle_reid2/generate_reid.py $R --video_base <clips> --label_base $OUT/labels/<labels> --save_base $OUT/datasets
# 3. Re-identification training (bagtricks_R50-ibn, 120 epochs from the VeRi-Wild model)
bash vehicle_reid2/run.sh $R "" 0 $OUT/reid/$R $OUT/datasets
# 4. Motion patterns from the training tracks
python track_cluster_updated.py --video_name <labels> --label_dir $OUT/labels --output_dir $OUT/motion_patterns \
    --k 0 --cluster_min_num 3
```

LEAP's distillation step (README step 7) was not used; extraction runs the undistilled detector.
`--cluster_min_num 3` reproduces the scored runs' cluster counts; LEAP's k-medoids starts from unseeded random centers,
so the clusters themselves differ between runs.

## Extraction on the test clips

```sh
python experiments/run/run_leap.py --config <workspace>/configs/<config> --weight <workspace>/models/yolov5/yolov5x.pt \
    --reid-weight <workspace>/logs --video-dir <clips>/test --work-dir <work> --export-prefix <out>/leap_rerun \
    [--motion-dir <workspace>/motion_patterns] [--retinanet] [--no-mask]
python experiments/eval/export_tracks.py --exporter experiments/eval/export-leap-outputs.py \
    --parsed-dir <work>/outputs/parsed_results/<full_name> --out <out>/leap_export.jsonl.gz [--clip-negative-starts]
```

B3D passes `--retinanet --weight <workspace>/models/retinanet/retinanet_b3d.pth`; Amsterdam and B3D pass `--motion-dir`.
`run_leap.py` repeats LEAP's `main.py` multi-video loop (all clips in one process, sorted file names, so LEAP's state
carries between clips as in the scored runs) and writes each clip's intervals, observations, and timers.
`export_tracks.py` rebuilds per-frame tracks with `export-leap-outputs.py`'s `reconstruct_tracks_json`.

## Adaptations Tetris applied

Each is opt-in; LEAP's own code is unchanged.

- Matched detector (`experiments/run/matched_detector.py`): labeling and extraction use Tetris's fine-tuned YOLOv5,
  with Tetris's inference defaults, mapping its single car class 0 to LEAP's COCO car class 2.
  Use `python experiments/run/matched_detector.py label <video_tracker args> --yolo_weight <checkpoint>` for step 1 and
  `python experiments/run/matched_detector.py <run_leap.py args>` for extraction.
- Motion-pattern IDs (`experiments/prep/motion_patterns.py`): BYTETracker restarts its IDs per clip, so this gives
  each clip's training tracks their own ID range before LEAP's clustering (replaces step 4).
- Reviewed regions (`experiments/configs/reviewed-regions/`): `stop_area` rectangles reviewed on the scene;
  every other key equals the workspace config.
- Unmasked CalDoT (`--no-mask`): disables LEAP's image mask.
- Negative starts (`--clip-negative-starts`): drops frame writes before frame 0 when exporting; pattern interpolation
  and LEAP's predicted intervals are unchanged.

Tetris's reported LEAP results use:

| Dataset | Version |
| --- | --- |
| Amsterdam | matched detector, motion-pattern IDs, reviewed regions, negative starts |
| CalDoT1, CalDoT2 | the same, unmasked |
| B3D1-4 | LEAP as run above, with the workspace's preparation |

Tetris's harness (`measurements-gpu/r1o2_leap_otif_rerun.py`, `r1o2_leap_matched.py`, and `r1o2_leap_regions_v2.py`
in the tetris repo) runs these steps under GNU time in a pinned container per step.
