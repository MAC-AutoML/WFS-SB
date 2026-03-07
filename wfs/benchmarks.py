"""Benchmark adapters shared by the WFS pipeline.

This module normalizes the metadata layout of different benchmarks so the
selection pipeline can operate on a common record structure. Each adapter knows
how to:

- read the raw annotation file,
- resolve the video path for one sample,
- find the relevance-score array saved during preprocessing, and
- write the final output item with selected keyframe indices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def similarity_key(feature_model: str) -> str:
    """Return the JSON key used to store similarity scores for a feature model.

    Args:
        feature_model: Short model identifier such as ``"blip2"`` or
            ``"clip"``.

    Returns:
        The key name expected inside ``similarity_scores.json``.
    """

    return f"{feature_model}_similarities"


def feature_filename(feature_model: str) -> str:
    """Return the filename used to persist frame-level visual features.

    Args:
        feature_model: Short model identifier used during preprocessing.

    Returns:
        The pickle filename that stores frame embeddings for the model.
    """

    return f"{feature_model}_vision_features.pkl"


@dataclass
class BenchmarkRecord:
    """Unified representation of one benchmark sample.

    Attributes:
        index: Index of the sample inside the raw annotation file.
        raw: Original JSON item. It is preserved so downstream code can emit the
            same fields back to the output file.
        feature_id: Directory name used to locate precomputed scores/features.
        video_path: Absolute or dataset-relative path to the source video file.
        question_index: Optional per-video question index used by benchmarks
            such as VideoMME where multiple questions share one video.
    """

    index: int
    raw: Dict[str, Any]
    feature_id: str
    video_path: Path
    question_index: Optional[int] = None


class BenchmarkAdapter:
    """Base adapter that converts a dataset into the WFS-friendly schema."""

    name: str = "base"

    def load_raw(self, questions_file: Path) -> List[Dict[str, Any]]:
        """Load and validate the raw benchmark annotation file.

        Args:
            questions_file: JSON file that stores the benchmark annotations.

        Returns:
            A list of raw annotation items.

        Raises:
            ValueError: If the JSON root is not a list.
        """

        with questions_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{self.name}: annotation file must be a list, got {type(data)}")
        return data

    def build_record(self, index: int, item: Dict[str, Any], dataset_root: Path) -> BenchmarkRecord:
        """Convert one raw item into a normalized record.

        Args:
            index: Position of the item inside the raw annotation list.
            item: Raw JSON object for one question or sample.
            dataset_root: Root directory of the benchmark dataset.

        Returns:
            A normalized ``BenchmarkRecord`` instance.
        """

        raise NotImplementedError

    def get_scores(
        self,
        scores_payload: Dict[str, Any],
        record: BenchmarkRecord,
        feature_model: str,
    ) -> Optional[np.ndarray]:
        """Extract the relevance-score vector for one record.

        Args:
            scores_payload: Parsed ``similarity_scores.json`` payload.
            record: Normalized record describing the current sample.
            feature_model: Short model identifier used to build the score key.

        Returns:
            A one-dimensional numpy array of frame relevance scores, or
            ``None`` when the payload does not contain a usable score sequence.
        """

        raise NotImplementedError

    def to_output_item(self, record: BenchmarkRecord, keyframe_indices: List[int]) -> Dict[str, Any]:
        """Attach selected keyframes to the original annotation item.

        Args:
            record: Original benchmark record for the sample.
            keyframe_indices: Final frame indices mapped back to the source
                video timeline.

        Returns:
            A shallow copy of the original annotation item with an added
            ``keyframe_indices`` field.
        """

        item = dict(record.raw)
        item["keyframe_indices"] = [int(i) for i in keyframe_indices]
        return item


class VideoMMEAdapter(BenchmarkAdapter):
    """Adapter for the VideoMME benchmark.

    VideoMME stores multiple question entries for the same video. The adapter
    therefore resolves both the shared video path and the per-question index so
    the correct score array can be recovered from the preprocessing output.
    """

    name = "videomme"

    def build_record(self, index: int, item: Dict[str, Any], dataset_root: Path) -> BenchmarkRecord:
        """Build a normalized VideoMME record.

        Args:
            index: Position inside the annotation file.
            item: Raw VideoMME question item.
            dataset_root: Root directory of the VideoMME dataset.

        Returns:
            A ``BenchmarkRecord`` with the shared video identifier and optional
            parsed question index.
        """

        question_index: Optional[int] = None
        qid = item.get("question_id", "")
        if isinstance(qid, str) and "-" in qid:
            try:
                question_index = int(qid.split("-")[-1]) - 1
            except ValueError:
                question_index = None

        video_file = f"{item['videoID']}.mp4"
        return BenchmarkRecord(
            index=index,
            raw=item,
            feature_id=str(item["video_id"]),
            video_path=dataset_root / "data" / video_file,
            question_index=question_index,
        )

    def get_scores(
        self,
        scores_payload: Dict[str, Any],
        record: BenchmarkRecord,
        feature_model: str,
    ) -> Optional[np.ndarray]:
        """Extract the score array for one VideoMME question.

        Args:
            scores_payload: Parsed preprocessing output for the video.
            record: Current VideoMME question record.
            feature_model: Model name used to derive the score key.

        Returns:
            The frame-level score vector for the matching question, or ``None``
            if the question cannot be matched.
        """

        key = similarity_key(feature_model)
        questions = scores_payload.get("questions", [])
        if not isinstance(questions, list):
            return None

        target_qidx = record.question_index
        if target_qidx is not None:
            for question_entry in questions:
                if question_entry.get("question_index") == target_qidx and key in question_entry:
                    return np.asarray(question_entry[key], dtype=float)

        # Fall back to raw ordering when a reliable question index is unavailable.
        if 0 <= record.index < len(questions):
            candidate = questions[record.index]
            if key in candidate:
                return np.asarray(candidate[key], dtype=float)
        return None


class LongVideoBenchAdapter(BenchmarkAdapter):
    """Adapter for LongVideoBench (LVB)."""

    name = "lvb"

    def build_record(self, index: int, item: Dict[str, Any], dataset_root: Path) -> BenchmarkRecord:
        """Build a normalized LVB record.

        Args:
            index: Position inside the annotation file.
            item: Raw LVB annotation item.
            dataset_root: Root directory of the LongVideoBench dataset.

        Returns:
            A normalized record that points to the LVB video file.
        """

        return BenchmarkRecord(
            index=index,
            raw=item,
            feature_id=str(index),
            video_path=dataset_root / "videos" / item["video_path"],
        )

    def get_scores(
        self,
        scores_payload: Dict[str, Any],
        record: BenchmarkRecord,
        feature_model: str,
    ) -> Optional[np.ndarray]:
        """Extract the single score array stored for an LVB sample."""

        key = similarity_key(feature_model)
        if key not in scores_payload:
            return None
        return np.asarray(scores_payload[key], dtype=float)


class MLVUAdapter(BenchmarkAdapter):
    """Adapter for the MLVU benchmark."""

    name = "mlvu"

    def build_record(self, index: int, item: Dict[str, Any], dataset_root: Path) -> BenchmarkRecord:
        """Build a normalized MLVU record.

        Args:
            index: Position inside the annotation file.
            item: Raw MLVU annotation item.
            dataset_root: Root directory of the MLVU dataset.

        Returns:
            A normalized record that points to the MLVU video file.
        """

        return BenchmarkRecord(
            index=index,
            raw=item,
            feature_id=str(item["question_id"]),
            video_path=dataset_root / "video" / item["video_name"],
        )

    def get_scores(
        self,
        scores_payload: Dict[str, Any],
        record: BenchmarkRecord,
        feature_model: str,
    ) -> Optional[np.ndarray]:
        """Extract the single score array stored for an MLVU sample."""

        key = similarity_key(feature_model)
        if key not in scores_payload:
            return None
        return np.asarray(scores_payload[key], dtype=float)


def create_adapter(benchmark: str) -> BenchmarkAdapter:
    """Create the adapter that matches a benchmark name.

    Args:
        benchmark: Benchmark identifier provided by the CLI.

    Returns:
        An initialized adapter for the requested benchmark.

    Raises:
        ValueError: If the benchmark name is unsupported.
    """

    normalized = benchmark.lower().strip()
    if normalized == "videomme":
        return VideoMMEAdapter()
    if normalized in {"lvb", "longvideobench"}:
        return LongVideoBenchAdapter()
    if normalized == "mlvu":
        return MLVUAdapter()
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def benchmark_defaults(benchmark: str) -> Dict[str, Any]:
    """Return default dataset paths and frame budgets for a benchmark.

    Args:
        benchmark: Benchmark identifier provided by the CLI.

    Returns:
        A dictionary containing default dataset root, annotation path,
        preprocessing directory, and target frame budget.

    Raises:
        ValueError: If the benchmark name is unsupported.
    """

    b = benchmark.lower().strip()
    if b == "videomme":
        return {
            "dataset_root": Path("datasets/videomme"),
            "questions_file": Path("datasets/videomme/videomme_json_file.json"),
            "features_dir": Path("datasets/videomme/blip2_features_and_scores"),
            "max_frames": 16,
        }
    if b in {"lvb", "longvideobench"}:
        return {
            "dataset_root": Path("datasets/longvideobench"),
            "questions_file": Path("datasets/longvideobench/lvb_val.json"),
            "features_dir": Path("datasets/longvideobench/blip2_features_and_scores"),
            "max_frames": 16,
        }
    if b == "mlvu":
        return {
            "dataset_root": Path("datasets/mlvu"),
            "questions_file": Path("datasets/mlvu/mlvu_dev.json"),
            "features_dir": Path("datasets/mlvu/blip2_features_and_scores"),
            "max_frames": 16,
        }
    raise ValueError(f"Unsupported benchmark: {benchmark}")
