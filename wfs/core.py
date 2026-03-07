"""Core Wavelet-based Frame Selection (WFS) components.

The implementation is organized into three small building blocks:

- ``WFSEventDetector`` finds temporal event boundaries from frame relevance
  scores using wavelet detail coefficients.
- ``WFSBudgetAllocator`` estimates how much of the frame budget should be spent
  on each detected event segment.
- ``WFSFrameSelector`` picks concrete frames inside each segment with an MMR
  strategy that balances relevance and diversity.

The ``WFS`` class ties these pieces together into a single end-to-end keyframe
selection routine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pywt
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity


class WFSEventDetector:
    """Detect event boundaries from a frame-level relevance sequence.

    Args:
        wavelet: Wavelet family name used by the discrete wavelet transform.
        mode: Signal padding mode passed to PyWavelets.
        height_factor: Multiplier applied to the standard deviation when
            building the adaptive peak-height threshold.
        prominence_factor: Fraction of the absolute-signal range used as the
            adaptive peak-prominence threshold.
    """

    def __init__(
        self,
        wavelet: str = "db4",
        mode: str = "symmetric",
        height_factor: float = 0.5,
        prominence_factor: float = 0.05,
    ) -> None:
        self.wavelet = wavelet
        self.mode = mode
        self.height_factor = height_factor
        self.prominence_factor = prominence_factor

    def decompose(self, relevance_scores: np.ndarray, level: int) -> List[np.ndarray]:
        """Run a discrete wavelet decomposition on frame relevance scores.

        Args:
            relevance_scores: One-dimensional frame-level relevance sequence.
            level: Target decomposition level.

        Returns:
            A list of approximation/detail coefficients returned by
            ``pywt.wavedec``.
        """

        return pywt.wavedec(relevance_scores, self.wavelet, level=level, mode=self.mode)

    def reconstruct_detail(
        self,
        coeffs: Sequence[np.ndarray],
        target_length: int,
        detail_level: int = 1,
    ) -> np.ndarray:
        """Reconstruct one detail band back into the original time domain.

        Args:
            coeffs: Wavelet coefficients returned by ``decompose``.
            target_length: Desired output length after inverse transform.
            detail_level: Detail band index to preserve. All other bands are
                zeroed before reconstruction.

        Returns:
            A reconstructed detail signal trimmed to ``target_length``.
        """

        coeffs_zero = [np.zeros_like(c) for c in coeffs]
        detail_level = int(np.clip(detail_level, 1, len(coeffs_zero) - 1))
        coeffs_zero[detail_level] = coeffs[detail_level]
        detail_signal = pywt.waverec(coeffs_zero, self.wavelet, mode=self.mode)
        return detail_signal[:target_length]

    def detect_peaks(self, detail_signal: np.ndarray, min_distance: int) -> np.ndarray:
        """Detect candidate event boundaries from a reconstructed detail signal.

        Args:
            detail_signal: Detail signal reconstructed in the original time
                domain.
            min_distance: Minimum number of frames allowed between two detected
                peaks.

        Returns:
            A numpy array containing peak indices.
        """

        abs_detail = np.abs(detail_signal)
        if len(abs_detail) == 0:
            return np.array([], dtype=int)

        mean_val = np.mean(abs_detail)
        std_val = np.std(abs_detail)
        height_threshold = mean_val + self.height_factor * std_val

        signal_range = np.max(abs_detail) - np.min(abs_detail)
        prominence_threshold = self.prominence_factor * signal_range

        peaks, _ = find_peaks(
            abs_detail,
            height=height_threshold,
            distance=max(1, int(min_distance)),
            prominence=prominence_threshold,
        )
        return peaks

    @staticmethod
    def create_segments(peaks: Sequence[int], total_frames: int) -> List[Tuple[int, int]]:
        """Convert ordered peak positions into half-open temporal segments.

        Args:
            peaks: Peak indices returned by ``detect_peaks``.
            total_frames: Total number of frames in the sampled sequence.

        Returns:
            A list of ``(start, end)`` frame spans.
        """

        boundaries = [0] + list(peaks) + [total_frames]
        return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


class WFSBudgetAllocator:
    """Distribute the frame budget across detected event segments.

    Args:
        w_duration: Weight for normalized segment duration.
        w_mean: Weight for the mean relevance score inside a segment.
        w_max: Weight for the maximum relevance score inside a segment.
        w_var: Weight for the normalized score variance inside a segment.
        strictness_factor: Controls how aggressively low-importance segments are
            filtered out.
        temperature: Softmax temperature used when turning importance scores
            into budget proportions.
        min_importance: Hard lower bound for the segment-importance threshold.
    """

    def __init__(
        self,
        w_duration: float = 0.3,
        w_mean: float = 0.4,
        w_max: float = 0.2,
        w_var: float = 0.1,
        strictness_factor: float = 1.5,
        temperature: float = 1.0,
        min_importance: float = 0.05,
    ) -> None:
        self.w_duration = w_duration
        self.w_mean = w_mean
        self.w_max = w_max
        self.w_var = w_var
        self.strictness_factor = strictness_factor
        self.temperature = temperature
        self.min_importance = min_importance

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute a temperature-scaled softmax distribution."""

        x = x / max(self.temperature, 1e-8)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def compute_importance(
        self,
        segment: Tuple[int, int],
        relevance_scores: np.ndarray,
        total_frames: int,
    ) -> float:
        """Compute the importance score of one segment.

        Args:
            segment: Half-open frame span ``(start, end)``.
            relevance_scores: Full frame-level relevance score array.
            total_frames: Length of the full score array.

        Returns:
            A scalar importance value used for segment filtering and budget
            allocation.
        """

        start, end = segment
        segment_scores = relevance_scores[start:end]
        if len(segment_scores) == 0:
            return 0.0

        duration_norm = (end - start) / max(total_frames, 1)
        mean_score = float(np.mean(segment_scores))
        max_score = float(np.max(segment_scores))
        var_segment = float(np.var(segment_scores))
        var_global = float(np.var(relevance_scores))
        var_norm = var_segment / (var_global + 1e-8)

        return (
            self.w_duration * duration_norm
            + self.w_mean * mean_score
            + self.w_max * max_score
            + self.w_var * var_norm
        )

    def filter_segments(
        self,
        segments: Sequence[Tuple[int, int]],
        importance_scores: Sequence[float],
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Remove segments whose importance is clearly too small.

        Args:
            segments: Candidate temporal segments.
            importance_scores: Importance value for each segment.

        Returns:
            A pair of filtered segments and their filtered importance scores.
            If all segments would be removed, the original inputs are returned to
            preserve robustness.
        """

        if len(segments) <= 3:
            return list(segments), list(importance_scores)

        mean_imp = float(np.mean(importance_scores))
        std_imp = float(np.std(importance_scores))
        adaptive_threshold = mean_imp - self.strictness_factor * std_imp
        threshold = max(self.min_importance, adaptive_threshold)

        kept_segments: List[Tuple[int, int]] = []
        kept_scores: List[float] = []
        for segment, score in zip(segments, importance_scores):
            if score >= threshold:
                kept_segments.append(segment)
                kept_scores.append(score)

        if not kept_segments:
            return list(segments), list(importance_scores)
        return kept_segments, kept_scores

    def allocate_budget(self, importance_scores: Sequence[float], total_budget: int) -> Dict[int, int]:
        """Allocate an integer frame budget to each surviving segment.

        Args:
            importance_scores: Importance values of the kept segments.
            total_budget: Total number of frames the pipeline must output.

        Returns:
            A mapping from filtered-segment index to allocated frame count.
        """

        if total_budget <= 0 or len(importance_scores) == 0:
            return {}

        proportions = self._softmax(np.asarray(importance_scores, dtype=float))
        raw_alloc = proportions * total_budget
        int_alloc = np.floor(raw_alloc).astype(int)

        allocation = {i: int(int_alloc[i]) for i in range(len(int_alloc))}

        # Distribute the remaining budget to the largest fractional remainders.
        remaining = int(total_budget - sum(allocation.values()))
        fractions = raw_alloc - int_alloc
        sorted_indices = np.argsort(fractions)[::-1]
        for i in range(remaining):
            allocation[int(sorted_indices[i])] += 1
        return allocation


class WFSFrameSelector:
    """Select concrete frames using an MMR-style relevance/diversity objective.

    Args:
        lambda_param: Trade-off between relevance and diversity. Values closer
            to ``1.0`` favor high-relevance frames, while lower values emphasize
            diversity.
    """

    def __init__(self, lambda_param: float = 0.5) -> None:
        self.lambda_param = lambda_param

    def select_from_segment(
        self,
        segment: Tuple[int, int],
        n_frames: int,
        relevance_scores: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Select frames from one segment with greedy MMR.

        Args:
            segment: Half-open segment boundaries ``(start, end)``.
            n_frames: Number of frames to pick from this segment.
            relevance_scores: Frame-level relevance scores.
            features: Optional frame features used to compute visual similarity.
                When absent, temporal distance is used as a cheap proxy.

        Returns:
            A list of selected frame indices in the sampled-frame space.
        """

        start, end = segment
        if end <= start or n_frames <= 0:
            return []

        segment_scores = relevance_scores[start:end]
        if len(segment_scores) == 0:
            return []

        # Always seed the segment with its most relevant local frame.
        anchor_local = int(np.argmax(segment_scores))
        anchor_global = start + anchor_local
        selected = [anchor_global]
        candidates = [start + i for i in range(len(segment_scores)) if i != anchor_local]

        while len(selected) < n_frames and candidates:
            mmr_scores = []
            for idx in candidates:
                relevance = relevance_scores[idx]
                if features is not None:
                    frame_feat = features[idx : idx + 1]
                    selected_feats = features[selected]
                    max_sim = float(np.max(cosine_similarity(frame_feat, selected_feats)[0]))
                else:
                    min_dist = min(abs(idx - s) for s in selected)
                    max_sim = float(np.exp(-min_dist / 10.0))

                mmr = self.lambda_param * relevance - (1.0 - self.lambda_param) * max_sim
                mmr_scores.append((idx, mmr))

            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best_idx)
            candidates.remove(best_idx)

        return selected

    def adjust_to_budget(
        self,
        selected_frames: Sequence[int],
        target_budget: int,
        relevance_scores: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Pad or trim the final selection so it matches the requested budget.

        Args:
            selected_frames: Frames chosen by segment-level selection.
            target_budget: Desired final number of frames.
            relevance_scores: Frame-level relevance scores used for fallback MMR
                expansion or relevance-based trimming.
            features: Optional frame feature matrix used for similarity-aware
                expansion.

        Returns:
            A sorted list of exactly ``target_budget`` frame indices when
            possible.
        """

        frames = sorted(set(int(i) for i in selected_frames))
        if target_budget <= 0:
            return []

        if len(frames) == target_budget:
            return frames

        if len(frames) < target_budget:
            candidates = [i for i in range(len(relevance_scores)) if i not in frames]
            while len(frames) < target_budget and candidates:
                mmr_scores = []
                for idx in candidates:
                    relevance = relevance_scores[idx]
                    if features is not None and len(frames) > 0:
                        frame_feat = features[idx : idx + 1]
                        selected_feats = features[frames]
                        max_sim = float(np.max(cosine_similarity(frame_feat, selected_feats)[0]))
                    elif len(frames) > 0:
                        min_dist = min(abs(idx - f) for f in frames)
                        max_sim = float(np.exp(-min_dist / 10.0))
                    else:
                        max_sim = 0.0
                    mmr = self.lambda_param * relevance - (1.0 - self.lambda_param) * max_sim
                    mmr_scores.append((idx, mmr))

                best_idx = max(mmr_scores, key=lambda x: x[1])[0]
                frames.append(best_idx)
                frames = sorted(frames)
                candidates.remove(best_idx)
        else:
            # If there are too many frames, keep the most relevant ones.
            scored = [(f, relevance_scores[f]) for f in frames]
            scored.sort(key=lambda x: x[1], reverse=True)
            frames = sorted([f for f, _ in scored[:target_budget]])

        return frames[:target_budget]


@dataclass
class WFSConfig:
    """Configuration container for the end-to-end WFS pipeline.

    Attributes:
        wavelet: Wavelet family used for event detection.
        lambda_param: MMR relevance/diversity trade-off.
        prominence_factor: Peak-prominence scaling factor.
        height_factor: Peak-height scaling factor.
        w_duration: Importance weight for segment duration.
        w_mean: Importance weight for mean segment relevance.
        w_max: Importance weight for max segment relevance.
        w_var: Importance weight for segment variance.
        strictness_factor: Strength of adaptive segment filtering.
        temperature: Softmax temperature for budget allocation.
    """

    wavelet: str = "db4"
    lambda_param: float = 0.5
    prominence_factor: float = 0.05
    height_factor: float = 0.5
    w_duration: float = 0.3
    w_mean: float = 0.4
    w_max: float = 0.2
    w_var: float = 0.1
    strictness_factor: float = 1.5
    temperature: float = 1.0


class WFS:
    """End-to-end Wavelet-based Frame Selection method."""

    def __init__(self, config: Optional[WFSConfig] = None) -> None:
        """Initialize the coordinated WFS components.

        Args:
            config: Optional configuration object. Default values are used when
                ``None`` is provided.
        """

        self.config = config or WFSConfig()
        self.event_detector = WFSEventDetector(
            wavelet=self.config.wavelet,
            height_factor=self.config.height_factor,
            prominence_factor=self.config.prominence_factor,
        )
        self.budget_allocator = WFSBudgetAllocator(
            w_duration=self.config.w_duration,
            w_mean=self.config.w_mean,
            w_max=self.config.w_max,
            w_var=self.config.w_var,
            strictness_factor=self.config.strictness_factor,
            temperature=self.config.temperature,
        )
        self.frame_selector = WFSFrameSelector(lambda_param=self.config.lambda_param)

    def select_keyframes(
        self,
        relevance_scores: np.ndarray,
        num_frames: int,
        dwt_level: int,
        min_peak_distance: int,
        features: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Select keyframes from a frame-level relevance curve.

        Args:
            relevance_scores: Frame-level relevance scores computed during
                preprocessing.
            num_frames: Target number of keyframes.
            dwt_level: DWT level used for event-boundary discovery.
            min_peak_distance: Minimum allowed distance between detected peaks.
            features: Optional visual features aligned with the score sequence.

        Returns:
            A list of selected frame indices in the sampled-frame space.
        """

        total_frames = len(relevance_scores)
        if total_frames == 0:
            return []
        if total_frames < num_frames:
            return list(range(total_frames))

        coeffs = self.event_detector.decompose(relevance_scores, dwt_level)
        detail_signal = self.event_detector.reconstruct_detail(coeffs, total_frames)
        peaks = self.event_detector.detect_peaks(detail_signal, min_peak_distance)
        segments = self.event_detector.create_segments(peaks, total_frames)

        # Preserve the original WAKS-style fallback when boundaries are weak.
        if len(segments) <= 1 or len(peaks) == 0:
            n_uniform = num_frames // 2
            n_top = num_frames - n_uniform
            uniform_indices = np.linspace(0, total_frames - 1, n_uniform, dtype=int)
            top_indices = np.argsort(relevance_scores)[::-1]

            top_selected: List[int] = []
            for idx in top_indices:
                if idx not in uniform_indices:
                    top_selected.append(int(idx))
                    if len(top_selected) >= n_top:
                        break

            merged = sorted(list(uniform_indices) + top_selected)
            return [int(i) for i in merged[:num_frames]]

        importance_scores = [
            self.budget_allocator.compute_importance(seg, relevance_scores, total_frames)
            for seg in segments
        ]
        valid_segments, valid_scores = self.budget_allocator.filter_segments(segments, importance_scores)
        allocation = self.budget_allocator.allocate_budget(valid_scores, num_frames)

        # Map scores from [0, 1] to [-1, 1] before the MMR stage so negative
        # relevance can more clearly penalize poor candidates.
        normalized_scores = 2.0 * relevance_scores - 1.0
        selected: List[int] = []
        for seg_idx, n_alloc in allocation.items():
            seg = valid_segments[seg_idx]
            selected.extend(
                self.frame_selector.select_from_segment(
                    segment=seg,
                    n_frames=n_alloc,
                    relevance_scores=normalized_scores,
                    features=features,
                )
            )

        return self.frame_selector.adjust_to_budget(
            selected_frames=selected,
            target_budget=num_frames,
            relevance_scores=normalized_scores,
            features=features,
        )


def compute_dwt_level(num_frames: int, wavelet: str = "db4", drift: int = 3) -> int:
    """Compute a safe and adaptive DWT level for a sequence length.

    Args:
        num_frames: Number of frames in the sampled sequence.
        wavelet: Wavelet family name.
        drift: Heuristic offset that reduces the raw ``log2`` estimate.

    Returns:
        A DWT level clipped to the range supported by PyWavelets.
    """

    if num_frames <= 1:
        return 1
    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(num_frames, wavelet_obj.dec_len)
    if max_level <= 1:
        return 1

    raw_level = int(np.floor(np.log2(num_frames) - drift))
    return int(np.clip(raw_level, 1, max_level))


def compute_min_peak_distance(num_frames: int, ratio: float = 0.02, absolute_min: int = 5) -> int:
    """Compute the minimum peak distance used by event detection.

    Args:
        num_frames: Number of frames in the sampled sequence.
        ratio: Relative minimum spacing as a fraction of sequence length.
        absolute_min: Hard lower bound on the distance.

    Returns:
        The final minimum peak spacing in frames.
    """

    return max(int(absolute_min), int(num_frames * ratio))
