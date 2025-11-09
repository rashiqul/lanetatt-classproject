# Datasets

## TuSimple

Lane detection dataset focused on highway scenes, mostly straight or slightly curved lanes.

- Resolution: ~1280×720 (depending on source)
- Splits: official train ≈ 6k images, test ≈ 2k (many repos use a smaller subset for speed)
- Evaluation: lane-wise accuracy. Sample y-coordinates; measure how close predicted x is to GT; compute accuracy per lane and average.
  - Note: LaneATT repos typically include a TuSimple evaluation script.

Why it’s good for us:
- Smaller and simpler
- Faster to prototype
- Good for debugging

Why it’s limited:
- Weak coverage of complex urban scenes
- Limited occlusion/intersection cases

## CULane

Much larger and more diverse dataset (urban, night, crowded, shadows).

- Size: ≈ 88k images across train/val/test and multiple scenarios
- Evaluation: F1 / precision / recall with IoU-like matching over lane segments
  - Report F1 at IoU=0.5 and per-scenario metrics (normal, crowded, night, no line, etc.)

Why it’s good for us:
- Better test of generalization
- Covers harder, more realistic cases

Why it’s painful:
- Heavy to download
- Requires preprocessing
- Stricter evaluation

## Conclusion

For fast iteration we will train and debug on a mini-TuSimple subset first, then, if time/compute permits, run a single CULane experiment to compare performance.