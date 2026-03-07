# Dataset Path Notes

This folder stores example JSON files for reproducible demos and ablations.

Included files:
- `videomme_f16.json`: VideoMME examples with 16 selected keyframes.
- `lvb_f16.json`: LongVideoBench examples with 16 selected keyframes.
- `mlvu_f16.json`: MLVU examples with 16 selected keyframes.

Expected full dataset layouts:
- `datasets/videomme/data/*.mp4`
- `datasets/longvideobench/videos/*.mp4`
- `datasets/mlvu/video/*.mp4`

Use `configs/dataset_paths.example.yaml` to map paths in your environment.
