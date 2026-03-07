# lmms-eval Diff and WFS Patches

This folder stores only `lmms-eval` diff artifacts for this project (documentation + patch + modified snapshots).
It is not a runnable `lmms-eval` checkout.

## Upstream Base

- Repository: `https://github.com/EvolvingLMMs-Lab/lmms-eval`
- Base commit: `bb1ebe76e7a942386c25c4664f902e0e59e8a401`
- Base commit date: `2025-09-29 09:41:22 +0800`
- Base commit message: `Add lemonade benchmark to the evaluation (#813)`

## Included Artifacts

- `lmms_eval_wfs.patch`: tracked-file diff + the added helper module.
- `modified_files/`: copied snapshots of all changed files for direct inspection.

## What Was Modified

1. CLI/task loading
- Added `--data_files` argument in `lmms_eval/__main__.py`.
- Passed `data_files` into `TaskManager` and task config override logic.
- Added `custom_load` path in `ConfigurableTask` (`datasets.load_dataset(path=..., data_files=...)`).

2. Logging/output naming
- `evaluation_tracker.save_results_aggregated(...)` supports `data_name`.
- Output filenames can be prefixed by custom frame-index json name.

3. Local dataset YAML/task updates
- VideoMME/LVB/MLVU task YAML files changed from online HF datasets to local JSON loading.
- Corresponding utils updated for local cache/video path behavior and prompt adjustments.
- LongVideoBench answer parsing logic was modified.

4. Keyframe-based model inference support
- Added `use_keyframe` support in `llava_vid`, `llava_onevision`, `internvl2`, `qwen2_vl`, `qwen2_5_vl`.
- Added new helper: `lmms_eval/models/model_utils/qwen2_5_vl_keyframe_vision_process.py`.

## Modified File List

Tracked modified files:
- `lmms_eval/__main__.py`
- `lmms_eval/api/task.py`
- `lmms_eval/loggers/evaluation_tracker.py`
- `lmms_eval/models/chat/qwen2_5_vl.py`
- `lmms_eval/models/simple/internvl2.py`
- `lmms_eval/models/simple/llava_onevision.py`
- `lmms_eval/models/simple/llava_vid.py`
- `lmms_eval/models/simple/qwen2_5_vl.py`
- `lmms_eval/models/simple/qwen2_vl.py`
- `lmms_eval/tasks/__init__.py`
- `lmms_eval/tasks/longvideobench/longvideobench_val_v.yaml`
- `lmms_eval/tasks/longvideobench/utils.py`
- `lmms_eval/tasks/mlvu/mlvu_dev.yaml`
- `lmms_eval/tasks/mlvu/utils.py`
- `lmms_eval/tasks/videomme/utils.py`
- `lmms_eval/tasks/videomme/videomme.yaml`
- `lmms_eval/tasks/videomme/videomme_w_subtitle.yaml`

New file:
- `lmms_eval/models/model_utils/qwen2_5_vl_keyframe_vision_process.py`

## Re-apply Patch to Fresh lmms-eval

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
git checkout bb1ebe76e7a942386c25c4664f902e0e59e8a401
git apply ../lmms-eval-diff/lmms_eval_wfs.patch
```

After applying, review task yaml/utils for your own local dataset paths.
