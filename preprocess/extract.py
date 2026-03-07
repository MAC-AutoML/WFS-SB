"""Unified preprocessing entry point for frame-level feature extraction.

The preprocessing stage samples frames from benchmark videos, computes a
query-conditioned relevance score for each sampled frame, and stores both the
score sequence and visual features for later use by WFS.

Supported feature backends:

- BLIP-2 image-text matching
- BLIP-1 image-text matching
- CLIP cosine similarity
- SigLIP sigmoid similarity

Supported benchmarks:

- VideoMME
- LongVideoBench (LVB)
- MLVU
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoProcessor,
    Blip2ForImageTextRetrieval,
    BlipForImageTextRetrieval,
    CLIPModel,
    CLIPProcessor,
)


# Default checkpoints used when the caller does not provide a custom model path.
MODEL_DEFAULTS = {
    "blip2": "Salesforce/blip2-itm-vit-g",
    "blip1": "Salesforce/blip-itm-base-coco",
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-so400m-patch14-384",
}

# Benchmark-specific dataset roots and annotation files.
BENCHMARK_DEFAULTS = {
    "videomme": {
        "dataset_root": Path("datasets/videomme"),
        "json_file": Path("datasets/videomme/videomme_json_file.json"),
    },
    "lvb": {
        "dataset_root": Path("datasets/longvideobench"),
        "json_file": Path("datasets/longvideobench/lvb_val.json"),
    },
    "mlvu": {
        "dataset_root": Path("datasets/mlvu"),
        "json_file": Path("datasets/mlvu/mlvu_dev.json"),
    },
}


def similarity_key(model_name: str) -> str:
    """Return the JSON field name used to store similarity scores.

    Args:
        model_name: Short feature-model identifier such as ``"blip2"``.

    Returns:
        The key expected in the JSON payload that stores frame-level similarity
        scores for the model.
    """

    return f"{model_name}_similarities"


def feature_filename(model_name: str) -> str:
    """Return the output filename used for frame-level visual features.

    Args:
        model_name: Short feature-model identifier.

    Returns:
        The pickle filename used to persist visual features.
    """

    return f"{model_name}_vision_features.pkl"


def parse_index_list(index_list: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated list of dataset indices.

    Args:
        index_list: String such as ``"1,4,9"`` or ``None``.

    Returns:
        A list of integers, or ``None`` when no explicit list is provided.
    """

    if not index_list:
        return None
    indices = []
    for token in index_list.split(","):
        token = token.strip()
        if token:
            indices.append(int(token))
    return indices


def _to_device(batch: Dict[str, torch.Tensor], device: str, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Move a processor output batch to the target device and dtype.

    Args:
        batch: Processor output dictionary containing tensors and metadata.
        device: Target device string such as ``"cuda"`` or ``"cpu"``.
        dtype: Floating-point dtype used for floating tensors.

    Returns:
        A new dictionary whose tensors are moved onto the requested device.
        Integer tensors keep their original dtype.
    """

    output: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            output[key] = value
            continue
        if value.dtype.is_floating_point:
            output[key] = value.to(device=device, dtype=dtype)
        else:
            output[key] = value.to(device=device)
    return output


class BaseExtractor:
    """Common interface shared by all feature extractors.

    Args:
        model_name: Short model identifier used for output naming.
        model_path: Hugging Face model path or local checkpoint path.
        device: Device string used for inference.
    """

    def __init__(self, model_name: str, model_path: str, device: str) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32

    def compute(self, frames: Sequence[Image.Image], query: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute frame-level similarities and visual features.

        Args:
            frames: Sampled video frames as PIL images.
            query: Text query used to score the frames.
            batch_size: Number of frames processed per forward pass.

        Returns:
            A tuple ``(similarities, features)`` where ``similarities`` is a
            one-dimensional tensor and ``features`` is a frame-aligned feature
            tensor.
        """

        raise NotImplementedError


class Blip2Extractor(BaseExtractor):
    """BLIP-2 image-text matching extractor."""

    def __init__(self, model_name: str, model_path: str, device: str) -> None:
        super().__init__(model_name, model_path, device)
        self.model = Blip2ForImageTextRetrieval.from_pretrained(model_path, torch_dtype=self.dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def compute(self, frames: Sequence[Image.Image], query: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score frames with BLIP-2 ITM and export image CLS features.

        Args:
            frames: Sampled video frames.
            query: Text query composed from the benchmark question.
            batch_size: Number of frames per inference batch.

        Returns:
            A tuple containing frame-level match probabilities and frame-level
            visual embeddings.
        """

        similarities = []
        features = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            batch_queries = [query] * len(batch_frames)
            inp = self.processor(
                images=batch_frames,
                text=batch_queries,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            inp = _to_device(inp, self.device, self.dtype)
            with torch.no_grad():
                outputs = self.model(**inp, use_image_text_matching_head=True)
                probs = torch.softmax(outputs.logits_per_image, dim=1)[:, 1]
                cls = outputs.image_embeds[:, 0, :]
            similarities.append(probs)
            features.append(cls)
        return torch.cat(similarities, dim=0), torch.cat(features, dim=0)


class Blip1Extractor(BaseExtractor):
    """BLIP-1 image-text matching extractor."""

    def __init__(self, model_name: str, model_path: str, device: str) -> None:
        super().__init__(model_name, model_path, device)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_path, torch_dtype=self.dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def compute(self, frames: Sequence[Image.Image], query: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score frames with BLIP-1 ITM and export CLS features.

        Args:
            frames: Sampled video frames.
            query: Text query composed from the benchmark question.
            batch_size: Number of frames per inference batch.

        Returns:
            A tuple containing frame-level match probabilities and frame-level
            hidden-state features.
        """

        similarities = []
        features = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            batch_queries = [query] * len(batch_frames)
            inp = self.processor(
                images=batch_frames,
                text=batch_queries,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            inp = _to_device(inp, self.device, self.dtype)
            with torch.no_grad():
                outputs = self.model(**inp, use_itm_head=True)
                probs = torch.softmax(outputs.itm_score, dim=1)[:, 1]
                cls = outputs.last_hidden_state[:, 0, :]
            similarities.append(probs)
            features.append(cls)
        return torch.cat(similarities, dim=0), torch.cat(features, dim=0)


class CLIPExtractor(BaseExtractor):
    """CLIP extractor that uses cosine similarity between image and text features."""

    def __init__(self, model_name: str, model_path: str, device: str) -> None:
        super().__init__(model_name, model_path, device)
        self.model = CLIPModel.from_pretrained(model_path, torch_dtype=self.dtype).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_path)

    def _image_features(self, frames: Sequence[Image.Image], batch_size: int) -> torch.Tensor:
        """Encode video frames into CLIP image embeddings.

        Args:
            frames: Sampled frames to encode.
            batch_size: Number of frames per forward pass.

        Returns:
            A tensor of frame-aligned CLIP image embeddings.
        """

        all_features = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            inp = self.processor(images=batch_frames, return_tensors="pt")
            inp = _to_device(inp, self.device, self.dtype)
            with torch.no_grad():
                feat = self.model.get_image_features(**inp)
            all_features.append(feat)
        return torch.cat(all_features, dim=0)

    def _text_features(self, query: str) -> torch.Tensor:
        """Encode the text query into a CLIP text embedding.

        Args:
            query: Question text used to score frames.

        Returns:
            A single CLIP text feature vector.
        """

        inp = self.processor(text=query, return_tensors="pt", truncation=True, padding=True)
        inp = _to_device(inp, self.device, self.dtype)
        with torch.no_grad():
            feat = self.model.get_text_features(**inp)
        return feat

    def compute(self, frames: Sequence[Image.Image], query: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CLIP cosine similarities for sampled frames.

        Args:
            frames: Sampled video frames.
            query: Text query composed from the benchmark question.
            batch_size: Number of frames per image-encoding batch.

        Returns:
            A tuple containing cosine similarities and raw image features.
        """

        image_feat = self._image_features(frames, batch_size=batch_size)
        text_feat = self._text_features(query)
        image_norm = F.normalize(image_feat, p=2, dim=-1)
        text_norm = F.normalize(text_feat, p=2, dim=-1)
        sims = torch.matmul(image_norm, text_norm.T).squeeze(-1)
        return sims, image_feat


class SigLIPExtractor(BaseExtractor):
    """SigLIP extractor that uses sigmoid-transformed logits as similarities."""

    def __init__(self, model_name: str, model_path: str, device: str) -> None:
        super().__init__(model_name, model_path, device)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=self.dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def compute(self, frames: Sequence[Image.Image], query: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute SigLIP similarities and image embeddings.

        Args:
            frames: Sampled video frames.
            query: Text query composed from the benchmark question.
            batch_size: Number of frames per inference batch.

        Returns:
            A tuple containing sigmoid similarity scores and visual embeddings.
        """

        similarities = []
        features = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            inp = self.processor(
                images=batch_frames,
                text=query,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
            )
            inp = _to_device(inp, self.device, self.dtype)
            with torch.no_grad():
                outputs = self.model(**inp)
                logits = outputs.logits_per_image
                probs = torch.sigmoid(logits).squeeze(-1)
                image_embeds = outputs.image_embeds
            similarities.append(probs)
            features.append(image_embeds)
        return torch.cat(similarities, dim=0), torch.cat(features, dim=0)


def build_extractor(model_name: str, model_path: str, device: str) -> BaseExtractor:
    """Instantiate the extractor that matches the requested feature model.

    Args:
        model_name: Short extractor identifier.
        model_path: Hugging Face model path or local checkpoint path.
        device: Device string used for inference.

    Returns:
        An initialized extractor instance.

    Raises:
        ValueError: If the model name is unsupported.
    """

    if model_name == "blip2":
        return Blip2Extractor(model_name, model_path, device)
    if model_name == "blip1":
        return Blip1Extractor(model_name, model_path, device)
    if model_name == "clip":
        return CLIPExtractor(model_name, model_path, device)
    if model_name == "siglip":
        return SigLIPExtractor(model_name, model_path, device)
    raise ValueError(f"Unsupported feature model: {model_name}")


def sample_frames(video_path: Path, sample_fps: float, num_threads: int) -> Tuple[List[int], List[Image.Image]]:
    """Sample frames from a video at an approximate target FPS.

    Args:
        video_path: Path to the source video.
        sample_fps: Desired sampling FPS. Values below the source FPS cause
            temporal downsampling.
        num_threads: Number of decoding threads passed to Decord.

    Returns:
        A tuple of ``(frame_indices, frames)`` where ``frame_indices`` are
        original frame numbers and ``frames`` are PIL images.
    """

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=num_threads)
    raw_fps = float(vr.get_avg_fps())
    if raw_fps <= 0:
        raw_fps = 1.0
    step = max(int(raw_fps / max(sample_fps, 1e-6)), 1)

    frame_indices = list(range(0, len(vr), step))
    if not frame_indices:
        frame_indices = [0]

    frames_np = vr.get_batch(frame_indices).asnumpy()
    frames = [Image.fromarray(frame) for frame in frames_np]
    return frame_indices, frames


@dataclass
class VideoQuestion:
    """One VideoMME question associated with a shared video.

    Attributes:
        question_index: Zero-based question index within the video group.
        question_id: Original benchmark question identifier.
        question: Natural-language question text.
        options: Candidate answers or options associated with the question.
        answer: Ground-truth answer in the original benchmark format.
        task_type: Optional task category string.
    """

    question_index: int
    question_id: str
    question: str
    options: List[str]
    answer: Any
    task_type: str


@dataclass
class VideoGroup:
    """Grouped VideoMME metadata for one source video.

    Attributes:
        video_id: Stable directory identifier used for preprocessing outputs.
        videoID: Original benchmark video identifier used for the file name.
        video_path: Path to the source video file.
        questions: All questions attached to the same video.
    """

    video_id: str
    videoID: str
    video_path: Path
    questions: List[VideoQuestion]


def load_videomme_groups(json_file: Path, dataset_root: Path) -> List[VideoGroup]:
    """Load VideoMME annotations and group questions by video.

    Args:
        json_file: Path to the VideoMME annotation JSON file.
        dataset_root: Root directory of the VideoMME dataset.

    Returns:
        A list of grouped video records, each containing all of its questions.
    """

    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    grouped: Dict[str, VideoGroup] = {}
    for item in data:
        vid = item["video_id"]
        if vid not in grouped:
            grouped[vid] = VideoGroup(
                video_id=vid,
                videoID=item["videoID"],
                video_path=dataset_root / "data" / f"{item['videoID']}.mp4",
                questions=[],
            )

        try:
            qidx = int(str(item["question_id"]).split("-")[-1]) - 1
        except Exception:
            qidx = len(grouped[vid].questions)

        grouped[vid].questions.append(
            VideoQuestion(
                question_index=qidx,
                question_id=item["question_id"],
                question=item["question"],
                options=item.get("options", []),
                answer=item.get("answer"),
                task_type=item.get("task_type", ""),
            )
        )

    return list(grouped.values())


def select_slices(total: int, start: int, end: int, index_list: Optional[List[int]]) -> List[int]:
    """Build the list of dataset indices to process.

    Args:
        total: Total number of available items.
        start: Inclusive start index when using range selection.
        end: Exclusive end index. Negative values mean "through the end".
        index_list: Optional explicit index list that overrides range
            selection.

    Returns:
        A list of valid indices to process.
    """

    if index_list is not None:
        return [idx for idx in index_list if 0 <= idx < total]

    if end < 0:
        end = total
    start = max(0, start)
    end = min(end, total)
    return list(range(start, end))


def process_videomme(
    groups: List[VideoGroup],
    selected_indices: List[int],
    extractor: BaseExtractor,
    output_dir: Path,
    model_name: str,
    sample_fps: float,
    batch_size: int,
    skip_existing: bool,
    max_retries: int,
    num_threads: int,
) -> Dict[str, Any]:
    """Run preprocessing for grouped VideoMME videos.

    Args:
        groups: Grouped VideoMME records, one per source video.
        selected_indices: Indices of groups that should be processed.
        extractor: Initialized feature extractor.
        output_dir: Root directory where preprocessing artifacts are written.
        model_name: Short model identifier used in output file names and keys.
        sample_fps: Approximate frame-sampling FPS.
        batch_size: Number of frames processed per model forward pass.
        skip_existing: Whether to skip items whose outputs already exist.
        max_retries: Number of retry attempts after the first failure.
        num_threads: Number of Decord decoding threads.

    Returns:
        A summary dictionary containing totals, failures, and timing statistics.
    """

    success = 0
    failed: List[str] = []
    total_read_time = 0.0
    total_model_time = 0.0

    for idx in tqdm(selected_indices, desc="Extract-VideoMME"):
        group = groups[idx]
        out_folder = output_dir / group.video_id
        score_path = out_folder / "similarity_scores.json"
        feat_path = out_folder / feature_filename(model_name)

        if skip_existing and score_path.exists() and feat_path.exists():
            continue

        if not group.video_path.exists():
            failed.append(group.video_id)
            continue

        done = False
        for attempt in range(max_retries + 1):
            try:
                read_start = time.time()
                frame_indices, frames = sample_frames(group.video_path, sample_fps=sample_fps, num_threads=num_threads)
                total_read_time += time.time() - read_start

                question_results = []
                first_feature: Optional[Any] = None

                # VideoMME stores several questions for one video, so we reuse
                # the sampled frames and run a separate text query per question.
                for q in group.questions:
                    query = q.question + " " + " ".join(q.options)
                    model_start = time.time()
                    similarities_t, features_t = extractor.compute(frames, query=query, batch_size=batch_size)
                    total_model_time += time.time() - model_start

                    # Visual features depend only on the frames, so storing the
                    # first computed feature tensor is sufficient.
                    if first_feature is None:
                        first_feature = features_t.detach().cpu().numpy()

                    question_results.append(
                        {
                            "question_index": q.question_index,
                            "options": q.options,
                            "query": query,
                            similarity_key(model_name): similarities_t.detach().cpu().tolist(),
                        }
                    )

                out_folder.mkdir(parents=True, exist_ok=True)
                with score_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "video_id": group.video_id,
                            "video_path": str(group.video_path),
                            "frame_indices": frame_indices,
                            "num_frames": len(frames),
                            "questions": question_results,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                with feat_path.open("wb") as f:
                    pickle.dump(first_feature, f)

                success += 1
                done = True
                break
            except Exception as exc:
                if attempt >= max_retries:
                    failed.append(group.video_id)
                else:
                    tqdm.write(f"[VideoMME] retry {attempt + 1}/{max_retries} for {group.video_id}: {exc}")

        if not done:
            continue

    return {
        "total": len(selected_indices),
        "success": success,
        "failed": failed,
        "total_read_time": total_read_time,
        "total_model_time": total_model_time,
    }


def process_lvb_or_mlvu(
    benchmark: str,
    data: List[Dict[str, Any]],
    selected_indices: List[int],
    dataset_root: Path,
    extractor: BaseExtractor,
    output_dir: Path,
    model_name: str,
    sample_fps: float,
    batch_size: int,
    skip_existing: bool,
    max_retries: int,
    num_threads: int,
) -> Dict[str, Any]:
    """Run preprocessing for LongVideoBench or MLVU.

    Args:
        benchmark: Either ``"lvb"`` or ``"mlvu"``.
        data: Raw benchmark annotation list.
        selected_indices: Indices of items that should be processed.
        dataset_root: Root directory of the benchmark dataset.
        extractor: Initialized feature extractor.
        output_dir: Root directory where preprocessing artifacts are written.
        model_name: Short model identifier used in output file names and keys.
        sample_fps: Approximate frame-sampling FPS.
        batch_size: Number of frames processed per model forward pass.
        skip_existing: Whether to skip items whose outputs already exist.
        max_retries: Number of retry attempts after the first failure.
        num_threads: Number of Decord decoding threads.

    Returns:
        A summary dictionary containing totals, failures, and timing statistics.
    """

    success = 0
    failed: List[str] = []
    total_read_time = 0.0
    total_model_time = 0.0

    for idx in tqdm(selected_indices, desc=f"Extract-{benchmark}"):
        item = data[idx]

        if benchmark == "lvb":
            output_id = str(idx)
            video_path = dataset_root / "videos" / item["video_path"]
            query = item["question"] + " " + " ".join(item.get("candidates", []))
        else:
            output_id = str(item["question_id"])
            video_path = dataset_root / "video" / item["video_name"]
            query = item["question"]

        out_folder = output_dir / output_id
        score_path = out_folder / "similarity_scores.json"
        feat_path = out_folder / feature_filename(model_name)

        if skip_existing and score_path.exists() and feat_path.exists():
            continue

        if not video_path.exists():
            failed.append(output_id)
            continue

        done = False
        for attempt in range(max_retries + 1):
            try:
                read_start = time.time()
                frame_indices, frames = sample_frames(video_path, sample_fps=sample_fps, num_threads=num_threads)
                total_read_time += time.time() - read_start

                model_start = time.time()
                similarities_t, features_t = extractor.compute(frames, query=query, batch_size=batch_size)
                total_model_time += time.time() - model_start

                out_folder.mkdir(parents=True, exist_ok=True)
                payload = {
                    "video_path": str(video_path),
                    "frame_indices": frame_indices,
                    "num_frames": len(frames),
                    "query": query,
                    similarity_key(model_name): similarities_t.detach().cpu().tolist(),
                }

                if benchmark == "lvb":
                    payload.update(
                        {
                            "index": output_id,
                            "question_id": item.get("id", ""),
                            "video_id": item.get("video_id", ""),
                            "question_category": item.get("question_category", ""),
                            "level": item.get("level", ""),
                            "topic_category": item.get("topic_category", ""),
                            "question": item.get("question", ""),
                            "candidates": item.get("candidates", []),
                            "answer": item.get("correct_choice", -1),
                        }
                    )
                else:
                    payload.update(
                        {
                            "question_id": item.get("question_id", ""),
                            "video_id": str(item.get("video_name", "")).replace(".mp4", ""),
                            "task_type": item.get("task_type", ""),
                            "question": item.get("question", ""),
                            "candidates": item.get("candidates", []),
                            "answer": item.get("answer", ""),
                        }
                    )

                with score_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                with feat_path.open("wb") as f:
                    pickle.dump(features_t.detach().cpu().numpy(), f)

                success += 1
                done = True
                break
            except Exception as exc:
                if attempt >= max_retries:
                    failed.append(output_id)
                else:
                    tqdm.write(f"[{benchmark}] retry {attempt + 1}/{max_retries} for {output_id}: {exc}")

        if not done:
            continue

    return {
        "total": len(selected_indices),
        "success": success,
        "failed": failed,
        "total_read_time": total_read_time,
        "total_model_time": total_model_time,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for preprocessing.

    Returns:
        A configured ``ArgumentParser`` with benchmark, model, sampling, and
        resume options.
    """

    parser = argparse.ArgumentParser(description="Unified feature extraction for VideoMME/LVB/MLVU")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["videomme", "lvb", "mlvu"],
        help="Benchmark to preprocess. This controls default dataset paths and output layout.",
    )
    parser.add_argument(
        "--feature_model",
        type=str,
        default="blip2",
        choices=["blip2", "blip1", "clip", "siglip"],
        help="Feature extractor backend used to compute frame similarities and visual embeddings.",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Dataset root directory. Uses the benchmark default when omitted.",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default=None,
        help="Path to the benchmark annotation JSON file. Uses the benchmark default when omitted.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where similarity scores and visual features are written.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional custom checkpoint path. Defaults to the predefined checkpoint for `feature_model`.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device such as `cuda`, `cuda:0`, or `cpu`. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=1.0,
        help="Approximate frame-sampling FPS used before similarity computation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of sampled frames processed per model forward pass.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=max(os.cpu_count() // 2, 1),
        help="Number of Decord decoding threads used when reading videos.",
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Inclusive start index when processing a contiguous slice of the dataset.",
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
        "--skip_existing",
        action="store_true",
        help="Skip samples whose similarity JSON and feature pickle already exist.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Number of retry attempts after an extraction failure.",
    )
    return parser


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill missing CLI arguments from benchmark and model defaults.

    Args:
        args: Parsed CLI namespace.

    Returns:
        The same namespace with missing fields populated.
    """

    defaults = BENCHMARK_DEFAULTS[args.benchmark]
    if args.dataset_root is None:
        args.dataset_root = str(defaults["dataset_root"])
    if args.json_file is None:
        args.json_file = str(defaults["json_file"])
    if args.output_dir is None:
        args.output_dir = str(Path(args.dataset_root) / f"{args.feature_model}_features_and_scores")
    if args.model_path is None:
        args.model_path = MODEL_DEFAULTS[args.feature_model]
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main() -> None:
    """CLI entry point for unified preprocessing."""

    parser = build_parser()
    args = parser.parse_args()
    args = resolve_defaults(args)

    dataset_root = Path(args.dataset_root)
    json_file = Path(args.json_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = build_extractor(args.feature_model, args.model_path, args.device)

    if args.benchmark == "videomme":
        groups = load_videomme_groups(json_file, dataset_root)
        selected = select_slices(
            total=len(groups),
            start=args.start_index,
            end=args.end_index,
            index_list=parse_index_list(args.index_list),
        )
        summary = process_videomme(
            groups=groups,
            selected_indices=selected,
            extractor=extractor,
            output_dir=output_dir,
            model_name=args.feature_model,
            sample_fps=args.sample_fps,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing,
            max_retries=args.max_retries,
            num_threads=args.num_threads,
        )
    else:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{args.benchmark}: expected a list JSON file")

        selected = select_slices(
            total=len(data),
            start=args.start_index,
            end=args.end_index,
            index_list=parse_index_list(args.index_list),
        )
        summary = process_lvb_or_mlvu(
            benchmark=args.benchmark,
            data=data,
            selected_indices=selected,
            dataset_root=dataset_root,
            extractor=extractor,
            output_dir=output_dir,
            model_name=args.feature_model,
            sample_fps=args.sample_fps,
            batch_size=args.batch_size,
            skip_existing=args.skip_existing,
            max_retries=args.max_retries,
            num_threads=args.num_threads,
        )

    summary.update(
        {
            "benchmark": args.benchmark,
            "feature_model": args.feature_model,
            "model_path": args.model_path,
            "device": args.device,
            "output_dir": str(output_dir),
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
