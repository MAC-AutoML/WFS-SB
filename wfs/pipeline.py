"""Command-line pipeline for running WFS on supported benchmarks.

This module glues together benchmark adapters, precomputed preprocessing
artifacts, and the core WFS selector. It also handles practical concerns such
as:

- selecting a subset of records by index,
- loading cached relevance scores and optional visual features,
- falling back to uniform sampling when data is incomplete, and
- writing benchmark-format output files.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm

from .benchmarks import (
    BenchmarkRecord,
    benchmark_defaults,
    create_adapter,
    feature_filename,
)
from .core import WFS, WFSConfig, compute_dwt_level, compute_min_peak_distance


def parse_index_list(index_list: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated index string.

    Args:
        index_list: String such as ``"0,3,10"`` or ``None``.

    Returns:
        A list of parsed integer indices, or ``None`` when the input is empty.
    """

    if not index_list:
        return None
    parsed: List[int] = []
    for token in index_list.split(","):
        token = token.strip()
        if not token:
            continue
        parsed.append(int(token))
    return parsed


def select_by_indices(
    raw_items: List[Dict[str, Any]],
    start: int,
    end: int,
    index_list: Optional[List[int]],
) -> List[tuple[int, Dict[str, Any]]]:
    """Select a subset of annotation items by range or explicit indices.

    Args:
        raw_items: Full list of raw annotation items.
        start: Inclusive start index used when ``index_list`` is not provided.
        end: Exclusive end index. Negative values mean "until the end".
        index_list: Optional explicit indices that override ``start``/``end``.

    Returns:
        A list of ``(index, item)`` pairs ready to be converted into benchmark
        records.
    """

    if index_list is not None:
        chosen = []
        for idx in index_list:
            if 0 <= idx < len(raw_items):
                chosen.append((idx, raw_items[idx]))
        return chosen

    if end < 0:
        end = len(raw_items)
    start = max(0, start)
    end = min(end, len(raw_items))
    return [(i, raw_items[i]) for i in range(start, end)]


def uniform_sample_from_video(video_path: Path, budget: int) -> List[int]:
    """Uniformly sample frame indices from a source video.

    This function is primarily used as a robust fallback when precomputed scores
    or features are missing.

    Args:
        video_path: Path to the source video.
        budget: Number of frame indices to return.

    Returns:
        A list of uniformly spaced frame indices. If the video cannot be read,
        a synthetic range is returned instead.
    """

    if budget <= 0:
        return []
    if video_path.exists():
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
            total = len(vr)
            if total > 0:
                return np.linspace(0, total - 1, budget, dtype=int).tolist()
        except Exception:
            pass
    return list(range(budget))


def _map_to_original_indices(selected: List[int], frame_indices: List[int], budget: int) -> List[int]:
    """Map sampled-frame indices back to the original video timeline.

    Args:
        selected: Indices selected in the sampled-frame space.
        frame_indices: Original frame numbers used during preprocessing.
        budget: Desired output length.

    Returns:
        A sorted list of original-frame indices. Missing positions are filled
        with uniformly spaced fallback indices so the output keeps a stable
        length.
    """

    if len(frame_indices) == 0:
        return list(range(budget))

    mapped: List[int] = []
    for idx in selected:
        if 0 <= idx < len(frame_indices):
            mapped.append(int(frame_indices[idx]))

    mapped = sorted(set(mapped))
    if len(mapped) < budget:
        fill_positions = np.linspace(0, len(frame_indices) - 1, budget, dtype=int).tolist()
        fill_frames = [int(frame_indices[p]) for p in fill_positions]
        merged = sorted(set(mapped + fill_frames))
        mapped = merged[:budget]

    if len(mapped) < budget:
        mapped = (mapped + [mapped[-1]] * (budget - len(mapped))) if mapped else list(range(budget))
    return mapped[:budget]


class FeatureStore:
    """Lazy loader for preprocessing artifacts.

    The store caches both JSON score payloads and feature arrays to avoid
    re-reading the same files when multiple benchmark records refer to the same
    video.

    Args:
        features_dir: Root directory that contains one subdirectory per sample.
    """

    def __init__(self, features_dir: Path) -> None:
        self.features_dir = features_dir
        self._score_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._feature_cache: Dict[str, Optional[np.ndarray]] = {}

    def load_scores(self, feature_id: str) -> Optional[Dict[str, Any]]:
        """Load ``similarity_scores.json`` for one sample.

        Args:
            feature_id: Sample identifier used as the artifact directory name.

        Returns:
            The parsed JSON payload, or ``None`` if the file is missing.
        """

        if feature_id in self._score_cache:
            return self._score_cache[feature_id]

        path = self.features_dir / feature_id / "similarity_scores.json"
        if not path.exists():
            self._score_cache[feature_id] = None
            return None

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self._score_cache[feature_id] = payload
        return payload

    def load_visual_features(self, feature_id: str, feature_model: str) -> Optional[np.ndarray]:
        """Load frame-level visual features for one sample.

        Args:
            feature_id: Sample identifier used as the artifact directory name.
            feature_model: Short model name used to build the feature filename.

        Returns:
            A numpy array of visual features, or ``None`` if the artifact is
            unavailable.
        """

        cache_key = f"{feature_id}:{feature_model}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        path = self.features_dir / feature_id / feature_filename(feature_model)
        if not path.exists():
            self._feature_cache[cache_key] = None
            return None

        with path.open("rb") as f:
            arr = pickle.load(f)
        if arr is None:
            self._feature_cache[cache_key] = None
            return None

        arr = np.asarray(arr)
        self._feature_cache[cache_key] = arr
        return arr


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the end-to-end WFS pipeline for the selected benchmark split.

    Args:
        args: Parsed CLI arguments. Important fields include benchmark paths,
            WFS hyperparameters, and output configuration.

    Returns:
        A summary dictionary describing how many records were processed, how
        many used fallback sampling, and where the result file was written.
    """

    adapter = create_adapter(args.benchmark)

    dataset_root = Path(args.dataset_root)
    questions_file = Path(args.questions_file)
    features_dir = Path(args.features_dir)

    raw_items = adapter.load_raw(questions_file)
    selected_raw = select_by_indices(
        raw_items=raw_items,
        start=args.start_index,
        end=args.end_index,
        index_list=parse_index_list(args.index_list),
    )

    records: List[BenchmarkRecord] = [
        adapter.build_record(index=i, item=item, dataset_root=dataset_root)
        for i, item in selected_raw
    ]

    wfs = WFS(
        WFSConfig(
            wavelet=args.wavelet,
            lambda_param=args.lambda_param,
            prominence_factor=args.prominence_factor,
            height_factor=args.height_factor,
            w_duration=args.w_duration,
            w_mean=args.w_mean,
            w_max=args.w_max,
            w_var=args.w_var,
            strictness_factor=args.strictness_factor,
            temperature=args.temperature,
        )
    )

    store = FeatureStore(features_dir=features_dir)
    results: List[Dict[str, Any]] = []
    missing_scores = 0
    fallback_count = 0
    success_count = 0

    for record in tqdm(records, desc=f"WFS-{adapter.name}"):
        scores_payload = store.load_scores(record.feature_id)
        if scores_payload is None:
            missing_scores += 1
            fallback_count += 1
            keyframes = uniform_sample_from_video(record.video_path, args.max_frames)
            results.append(adapter.to_output_item(record, keyframes))
            continue

        relevance_scores = adapter.get_scores(scores_payload, record, args.feature_model)
        frame_indices = scores_payload.get("frame_indices", [])

        # When relevance data is unavailable, keep the pipeline robust by using
        # uniform temporal sampling instead of failing hard.
        if relevance_scores is None or len(relevance_scores) == 0:
            fallback_count += 1
            keyframes = uniform_sample_from_video(record.video_path, args.max_frames)
            results.append(adapter.to_output_item(record, keyframes))
            continue

        if not isinstance(frame_indices, list) or len(frame_indices) == 0:
            frame_indices = list(range(len(relevance_scores)))

        if len(relevance_scores) < args.max_frames:
            fallback_count += 1
            keyframes = uniform_sample_from_video(record.video_path, args.max_frames)
            results.append(adapter.to_output_item(record, keyframes))
            continue

        dwt_level = compute_dwt_level(len(relevance_scores), wavelet=args.wavelet, drift=args.drift_level)
        min_peak_distance = compute_min_peak_distance(
            len(relevance_scores),
            ratio=args.min_distance_ratio,
            absolute_min=args.min_distance_absolute,
        )

        features = None
        if not args.no_visual_features:
            features = store.load_visual_features(record.feature_id, args.feature_model)

        selected = wfs.select_keyframes(
            relevance_scores=np.asarray(relevance_scores, dtype=float),
            num_frames=args.max_frames,
            dwt_level=dwt_level,
            min_peak_distance=min_peak_distance,
            features=features,
        )
        keyframes = _map_to_original_indices(selected, frame_indices, args.max_frames)

        results.append(adapter.to_output_item(record, keyframes))
        success_count += 1

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path("outputs") / adapter.name / f"WFS_{adapter.name}_{args.feature_model}_{args.max_frames}f.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return {
        "benchmark": adapter.name,
        "num_records": len(records),
        "success": success_count,
        "fallback": fallback_count,
        "missing_scores": missing_scores,
        "output_path": str(output_path),
        "wfs_config": asdict(wfs.config),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the WFS pipeline.

    Returns:
        A configured ``ArgumentParser`` with dataset, selection, and WFS
        hyperparameter options.
    """

    parser = argparse.ArgumentParser(description="Run unified WFS pipeline on VideoMME/LVB/MLVU")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["videomme", "lvb", "mlvu"],
        help="Benchmark to process. This controls the adapter, default paths, and output format.",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Dataset root directory. When omitted, the benchmark-specific default path is used.",
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default=None,
        help="Path to the benchmark annotation JSON file. Defaults depend on the benchmark.",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
        help="Directory that contains preprocessing outputs such as similarity scores and visual features.",
    )
    parser.add_argument(
        "--feature_model",
        type=str,
        default="blip2",
        choices=["blip2", "blip1", "clip", "siglip"],
        help="Feature model name used to locate the correct score and feature files.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Target number of output keyframes per sample. Defaults to the benchmark preset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output JSON path. If not set, a benchmark-specific path under `outputs/` is used.",
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Inclusive start index when processing a contiguous slice of the annotation file.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=-1,
        help="Exclusive end index for slice processing. Use -1 to process through the end.",
    )
    parser.add_argument(
        "--index_list",
        type=str,
        default=None,
        help="Comma-separated explicit indices to process, overriding `start_index` and `end_index`.",
    )
    parser.add_argument(
        "--no_visual_features",
        action="store_true",
        help="Disable visual-feature loading and use temporal distance as the diversity proxy.",
    )

    parser.add_argument(
        "--wavelet",
        type=str,
        default="db4",
        help="Wavelet family used for discrete wavelet decomposition during event detection.",
    )
    parser.add_argument(
        "--drift_level",
        type=int,
        default=3,
        help="Heuristic offset applied when estimating a safe DWT level from the sequence length.",
    )
    parser.add_argument(
        "--height_factor",
        type=float,
        default=0.5,
        help="Multiplier for the adaptive peak-height threshold in wavelet-detail peak detection.",
    )
    parser.add_argument(
        "--prominence_factor",
        type=float,
        default=0.05,
        help="Multiplier for the adaptive peak-prominence threshold in event detection.",
    )
    parser.add_argument(
        "--min_distance_ratio",
        type=float,
        default=0.02,
        help="Minimum peak spacing expressed as a fraction of the sampled sequence length.",
    )
    parser.add_argument(
        "--min_distance_absolute",
        type=int,
        default=5,
        help="Absolute lower bound for the minimum distance between detected event peaks.",
    )

    parser.add_argument(
        "--w_duration",
        type=float,
        default=0.4,
        help="Segment-importance weight for normalized duration.",
    )
    parser.add_argument(
        "--w_mean",
        type=float,
        default=0.2,
        help="Segment-importance weight for mean relevance score.",
    )
    parser.add_argument(
        "--w_max",
        type=float,
        default=0.3,
        help="Segment-importance weight for maximum relevance score.",
    )
    parser.add_argument(
        "--w_var",
        type=float,
        default=0.1,
        help="Segment-importance weight for relevance-score variance.",
    )
    parser.add_argument(
        "--strictness_factor",
        type=float,
        default=1.2,
        help="Controls how aggressively low-importance segments are filtered out.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature used when converting importance scores into segment budgets.",
    )
    parser.add_argument(
        "--lambda_param",
        type=float,
        default=0.5,
        help="MMR trade-off between frame relevance and diversity during keyframe selection.",
    )
    return parser


def resolve_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Fill missing path and budget arguments from benchmark presets.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The same namespace with benchmark-specific default values populated.
    """

    defaults = benchmark_defaults(args.benchmark)
    if args.dataset_root is None:
        args.dataset_root = str(defaults["dataset_root"])
    if args.questions_file is None:
        args.questions_file = str(defaults["questions_file"])
    if args.features_dir is None:
        args.features_dir = str(defaults["features_dir"])
    if args.max_frames is None:
        args.max_frames = int(defaults["max_frames"])
    return args


def main() -> None:
    """CLI entry point for the WFS pipeline."""

    parser = build_parser()
    args = parser.parse_args()
    args = resolve_default_paths(args)
    summary = run_pipeline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
