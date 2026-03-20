"""LPRADO-005: Sample Weights by Uniqueness — de-bias overlapping labels.

Implements sample weighting from Lopez de Prado's *Advances in Financial
Machine Learning* (Chapter 4).

Problem: When labels span multiple bars (e.g., triple-barrier labels with
a 10-bar horizon), samples overlap in time. A bar at time t may contribute
to labels for samples starting at t-9, t-8, ..., t. This overlap means
samples are not independent — training on them without correction leads
to overfitting and inflated cross-validation scores.

Solution: Weight each sample by its *uniqueness*, defined as the inverse
of the average number of concurrent labels during the sample's lifespan.
Combine with time-decay weights to emphasize recent observations.

Usage:
    sw = SampleWeighting()
    uniq = sw.compute_uniqueness_weights(start_times, end_times)
    decay = sw.compute_time_decay_weights(n_samples=len(uniq), decay=0.5)
    combined = sw.combine_weights(uniq, decay)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SampleWeighting:
    """Compute sample weights based on uniqueness and time decay.

    Designed for use with triple-barrier or any time-span label method
    where samples have overlapping observation periods.
    """

    # ------------------------------------------------------------------
    # Uniqueness weights
    # ------------------------------------------------------------------

    @staticmethod
    def compute_uniqueness_weights(
        labels_start: pd.Series | np.ndarray,
        labels_end: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """Compute sample uniqueness weights based on label concurrency.

        For each sample i with label spanning [start_i, end_i], the
        uniqueness is:
            u_i = mean(1 / c_t) for t in [start_i, end_i]

        where c_t is the number of labels active at time t.

        Args:
            labels_start: Start times (or indices) of each label span.
            labels_end: End times (or indices) of each label span.
                        Must be same length as labels_start.

        Returns:
            1-D array of uniqueness weights in (0, 1], same length as input.
            Weight = 1.0 means the sample has no overlap with any other sample.
        """
        starts = np.asarray(labels_start)
        ends = np.asarray(labels_end)
        n_samples = len(starts)

        if n_samples == 0:
            return np.array([], dtype=np.float64)

        if len(ends) != n_samples:
            raise ValueError(
                f"labels_start ({n_samples}) and labels_end ({len(ends)}) "
                f"must have the same length"
            )

        # Handle both integer indices and datetime-like inputs
        if hasattr(starts[0], 'timestamp') or isinstance(starts[0], (pd.Timestamp, np.datetime64)):
            # Convert to integer positions for the concurrency matrix
            return SampleWeighting._uniqueness_datetime(starts, ends, n_samples)
        else:
            return SampleWeighting._uniqueness_integer(
                starts.astype(np.int64), ends.astype(np.int64), n_samples,
            )

    @staticmethod
    def _uniqueness_integer(
        starts: np.ndarray,
        ends: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Compute uniqueness for integer-indexed label spans."""
        t_min = int(starts.min())
        t_max = int(ends.max())
        span = t_max - t_min + 1

        # Concurrency array: how many labels are active at each time step
        concurrency = np.zeros(span, dtype=np.int32)
        for i in range(n_samples):
            s = int(starts[i]) - t_min
            e = int(ends[i]) - t_min + 1
            concurrency[s:e] += 1

        # Uniqueness for each sample
        weights = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            s = int(starts[i]) - t_min
            e = int(ends[i]) - t_min + 1
            c_slice = concurrency[s:e]

            # Avoid division by zero (shouldn't happen but be safe)
            valid = c_slice[c_slice > 0]
            if len(valid) > 0:
                weights[i] = float(np.mean(1.0 / valid))
            else:
                weights[i] = 1.0

        return weights

    @staticmethod
    def _uniqueness_datetime(
        starts: np.ndarray,
        ends: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Compute uniqueness for datetime-indexed label spans.

        Uses a sweep-line algorithm for efficiency with datetime indices.
        """
        # Convert to pandas Timestamps for uniform handling
        starts_ts = pd.to_datetime(starts)
        ends_ts = pd.to_datetime(ends)

        # Collect all unique time points
        all_times = np.unique(np.concatenate([starts_ts.values, ends_ts.values]))
        all_times.sort()

        # Map each time to an integer index
        time_to_idx = {t: i for i, t in enumerate(all_times)}
        span = len(all_times)

        # Concurrency at each unique time point
        concurrency = np.zeros(span, dtype=np.int32)
        sample_ranges = []

        for i in range(n_samples):
            s_idx = time_to_idx[starts_ts[i]]
            e_idx = time_to_idx[ends_ts[i]]
            sample_ranges.append((s_idx, e_idx))
            concurrency[s_idx: e_idx + 1] += 1

        # Uniqueness
        weights = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            s_idx, e_idx = sample_ranges[i]
            c_slice = concurrency[s_idx: e_idx + 1]
            valid = c_slice[c_slice > 0]
            if len(valid) > 0:
                weights[i] = float(np.mean(1.0 / valid))
            else:
                weights[i] = 1.0

        return weights

    # ------------------------------------------------------------------
    # Time-decay weights
    # ------------------------------------------------------------------

    @staticmethod
    def compute_time_decay_weights(
        n_samples: int,
        decay: float = 0.5,
    ) -> np.ndarray:
        """Compute time-decay weights that emphasize recent observations.

        Produces a linearly decaying weight schedule from `decay` (oldest
        sample) to 1.0 (most recent sample). The decay parameter controls
        how much weight is given to old samples relative to new ones.

        Args:
            n_samples: Number of samples.
            decay: Weight of the oldest sample relative to newest.
                   decay=1.0 gives uniform weights (no decay).
                   decay=0.5 gives oldest sample half the weight of newest.
                   decay=0.0 gives oldest sample zero weight.
                   Must be in [0, 1].

        Returns:
            1-D array of weights, monotonically increasing from decay to 1.0.
        """
        if n_samples <= 0:
            return np.array([], dtype=np.float64)

        decay = max(0.0, min(1.0, decay))

        if n_samples == 1:
            return np.array([1.0])

        # Linear interpolation from decay to 1.0
        weights = np.linspace(decay, 1.0, n_samples)

        return weights

    # ------------------------------------------------------------------
    # Combined weights
    # ------------------------------------------------------------------

    @staticmethod
    def combine_weights(
        uniqueness: np.ndarray,
        time_decay: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Combine uniqueness and time-decay weights multiplicatively.

        The combined weight for sample i is:
            w_i = uniqueness_i * time_decay_i

        Optionally normalizes so weights sum to n_samples (preserving
        effective sample size semantics for ML frameworks).

        Args:
            uniqueness: Uniqueness weights from compute_uniqueness_weights.
            time_decay: Time-decay weights from compute_time_decay_weights.
            normalize: If True, normalize so sum = n_samples. Default True.

        Returns:
            Combined weight array.

        Raises:
            ValueError: If arrays have different lengths.
        """
        if len(uniqueness) != len(time_decay):
            raise ValueError(
                f"Weight arrays must have same length: "
                f"uniqueness={len(uniqueness)}, time_decay={len(time_decay)}"
            )

        if len(uniqueness) == 0:
            return np.array([], dtype=np.float64)

        combined = uniqueness * time_decay

        if normalize:
            total = combined.sum()
            if total > 1e-10:
                combined = combined * (len(combined) / total)

        return combined

    # ------------------------------------------------------------------
    # Convenience: full pipeline
    # ------------------------------------------------------------------

    def compute_sample_weights(
        self,
        labels_start: pd.Series | np.ndarray,
        labels_end: pd.Series | np.ndarray,
        decay: float = 0.5,
    ) -> np.ndarray:
        """Full pipeline: uniqueness + time decay -> combined weights.

        Args:
            labels_start: Start of each label span.
            labels_end: End of each label span.
            decay: Time-decay parameter. Default 0.5.

        Returns:
            Combined, normalized sample weights.
        """
        uniqueness = self.compute_uniqueness_weights(labels_start, labels_end)
        time_decay = self.compute_time_decay_weights(len(uniqueness), decay)
        combined = self.combine_weights(uniqueness, time_decay)

        logger.info(
            f"Sample weights computed: n={len(combined)}, "
            f"mean_uniqueness={uniqueness.mean():.3f}, "
            f"min_weight={combined.min():.3f}, max_weight={combined.max():.3f}"
        )

        return combined

    # ------------------------------------------------------------------
    # Sequential bootstrap helper
    # ------------------------------------------------------------------

    @staticmethod
    def sequential_bootstrap_indices(
        labels_start: pd.Series | np.ndarray,
        labels_end: pd.Series | np.ndarray,
        n_draws: int | None = None,
    ) -> np.ndarray:
        """Generate bootstrap sample indices weighted by average uniqueness.

        The sequential bootstrap (Lopez de Prado) draws samples one at a time,
        re-computing uniqueness weights conditional on previously drawn samples.
        This produces bootstrap samples with lower redundancy than IID bootstrap.

        Args:
            labels_start: Start of each label span.
            labels_end: End of each label span.
            n_draws: Number of bootstrap draws. Default = n_samples.

        Returns:
            Array of sampled indices (with replacement, weighted by uniqueness).
        """
        starts = np.asarray(labels_start)
        ends = np.asarray(labels_end)
        n_samples = len(starts)

        if n_draws is None:
            n_draws = n_samples

        if n_samples == 0:
            return np.array([], dtype=np.int64)

        # Convert to integer indices if datetime
        if hasattr(starts[0], 'timestamp') or isinstance(starts[0], (pd.Timestamp, np.datetime64)):
            starts_dt = pd.to_datetime(starts)
            ends_dt = pd.to_datetime(ends)
            all_times = np.sort(np.unique(np.concatenate([starts_dt.values, ends_dt.values])))
            time_map = {t: i for i, t in enumerate(all_times)}
            int_starts = np.array([time_map[t] for t in starts_dt], dtype=np.int64)
            int_ends = np.array([time_map[t] for t in ends_dt], dtype=np.int64)
        else:
            int_starts = starts.astype(np.int64)
            int_ends = ends.astype(np.int64)

        t_min = int(int_starts.min())
        t_max = int(int_ends.max())
        span = t_max - t_min + 1

        drawn_indices = np.zeros(n_draws, dtype=np.int64)
        concurrency = np.zeros(span, dtype=np.int32)

        for draw in range(n_draws):
            # Compute uniqueness given current concurrency
            probs = np.zeros(n_samples, dtype=np.float64)
            for i in range(n_samples):
                s = int(int_starts[i]) - t_min
                e = int(int_ends[i]) - t_min + 1
                c_slice = concurrency[s:e] + 1  # +1 for this candidate
                probs[i] = float(np.mean(1.0 / c_slice))

            # Normalize to probability distribution
            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                probs = np.ones(n_samples) / n_samples

            # Draw one sample
            idx = np.random.choice(n_samples, p=probs)
            drawn_indices[draw] = idx

            # Update concurrency
            s = int(int_starts[idx]) - t_min
            e = int(int_ends[idx]) - t_min + 1
            concurrency[s:e] += 1

        return drawn_indices
