"""Pumping event detection utilities.

This module combines several heuristics that work well on groundwater
time-series contaminated by pumping:

- abrupt dry-day drops (`sigma`)
- rolling-baseline level shifts (`rolling_baseline`)
- periodic pumping signatures (`fourier`)

Note on naming
--------------
The rolling-baseline method was previously labelled "pelt" in method
lists and API arguments.  This was misleading — the algorithm is NOT
the Pruned Exact Linear Time (PELT) change-point detector of Killick
et al. (2012, JASA).  It is a simpler heuristic that compares water
levels against a rolling mean and flags sustained deviations on dry
days.  As of v32 the parameter has been renamed from ``pelt_penalty``
to ``baseline_penalty`` to eliminate the misnomer.  The old keyword
argument ``pelt_penalty`` is still accepted for backward compatibility
but is deprecated and will be removed in a future release.
The legacy method alias ``pelt`` is also still accepted but deprecated.

References
----------
Killick, R., Fearnhead, P. & Eckley, I.A. (2012). Optimal detection
    of changepoints with a linear computational cost.  Journal of the
    American Statistical Association, 107(500), 1590-1598.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DetectionResult:
    """Container for pumping detection outputs."""

    pump_mask: np.ndarray
    confidence: np.ndarray
    method_masks: dict
    n_pump_days: int
    pump_fraction: float
    drop_events: List[dict]
    dominant_period: Optional[float] = None
    diagnostics: dict = field(default_factory=dict)


class PumpingDetector:
    """Detect pumping intervals from water-level and rainfall series."""

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        sigma_drop: float = 2.5,
        sigma_run: float = 0.8,
        run_min_days: int = 3,
        baseline_penalty: float = 3.0,
        fourier_threshold: float = 0.15,
        rainfall_cutoff: float = 1.0,
        buffer_days: int = 2,
        min_confidence: float = 0.4,
        # ── deprecated alias ──
        pelt_penalty: Optional[float] = None,
    ):
        self.methods = methods or ["sigma", "rolling_baseline"]
        self.sigma_drop = sigma_drop
        self.sigma_run = sigma_run
        self.run_min_days = run_min_days
        # v32: renamed pelt_penalty → baseline_penalty (misnomer fix).
        # Accept old kwarg for backward compat, with deprecation warning.
        if pelt_penalty is not None:
            import warnings
            warnings.warn(
                "pelt_penalty is deprecated; use baseline_penalty instead.",
                DeprecationWarning, stacklevel=2,
            )
            baseline_penalty = pelt_penalty
        self.baseline_penalty = baseline_penalty
        self.fourier_threshold = fourier_threshold
        self.rainfall_cutoff = rainfall_cutoff
        self.buffer_days = buffer_days
        self.min_confidence = min_confidence

    def detect(
        self,
        dates: pd.DatetimeIndex,
        water_level: np.ndarray,
        rainfall: np.ndarray,
        known_pump_times: Optional[List[Tuple]] = None,
    ) -> DetectionResult:
        """Run configured detection methods and merge their outputs."""
        n = len(water_level)
        wl = np.array(water_level, dtype=float)
        po = np.array(rainfall, dtype=float)

        dry = po < self.rainfall_cutoff
        dh = np.concatenate([[0.0], np.diff(wl)])

        dry_dh = dh[dry & ~np.isnan(wl)]
        sig = np.nanstd(dry_dh) if len(dry_dh) > 5 else 0.01
        sig = max(sig, 0.003)

        method_masks = {}
        confidence = np.zeros(n)

        if "sigma" in self.methods:
            m1, events = self._method_sigma(wl, dh, dry, sig)
            method_masks["sigma"] = m1
            confidence += m1.astype(float) * 0.45
        else:
            events = []

        if "rolling_baseline" in self.methods or "pelt" in self.methods:
            m2 = self._method_rolling_baseline(wl, dry, sig)
            method_masks["rolling_baseline"] = m2
            confidence += m2.astype(float) * 0.35

        dominant_period = None
        if "fourier" in self.methods:
            m3, dominant_period = self._method_fourier(wl, sig)
            method_masks["fourier"] = m3
            confidence += m3.astype(float) * 0.20

        if known_pump_times:
            m_known = self._apply_known_pump(dates, known_pump_times, n)
            method_masks["known"] = m_known
            confidence[m_known] = 1.0

        max_conf = confidence.max()
        if max_conf > 0:
            confidence = confidence / max_conf * np.clip(max_conf, 0, 1)

        raw_mask = confidence >= self.min_confidence
        pump_mask = self._apply_buffer(raw_mask, self.buffer_days, n)

        diagnostics = {
            "sigma_used": sig,
            "dry_days": int(dry.sum()),
            "sigma_drop_threshold_m": float(-self.sigma_drop * sig),
            "methods_applied": self.methods.copy(),
        }

        return DetectionResult(
            pump_mask=pump_mask,
            confidence=confidence,
            method_masks=method_masks,
            n_pump_days=int(pump_mask.sum()),
            pump_fraction=float(pump_mask.sum() / n),
            drop_events=self._describe_events(dates, wl, pump_mask),
            dominant_period=dominant_period,
            diagnostics=diagnostics,
        )

    def _method_sigma(
        self,
        wl: np.ndarray,
        dh: np.ndarray,
        dry: np.ndarray,
        sig: float,
    ) -> Tuple[np.ndarray, List[dict]]:
        """Detect abrupt spikes and sustained declines on dry days."""
        n = len(wl)
        mask = np.zeros(n, dtype=bool)
        events: List[dict] = []

        th_spike = -self.sigma_drop * sig
        spike_idx = np.where(dry & (dh < th_spike) & ~np.isnan(wl))[0]
        for ii in spike_idx:
            s = max(0, ii - 1)
            e = min(n, ii + 4)
            mask[s:e] = True
            events.append(
                {
                    "type": "spike_drop",
                    "index": int(ii),
                    "magnitude_m": float(dh[ii]),
                    "threshold_m": float(th_spike),
                }
            )

        th_run = -self.sigma_run * sig
        neg_dry = dry & (dh < th_run) & ~np.isnan(wl)
        i = 0
        while i < n:
            if neg_dry[i]:
                j = i
                while j < n and neg_dry[j]:
                    j += 1
                run_len = j - i
                if run_len >= self.run_min_days:
                    s = max(0, i - 1)
                    e = min(n, j + 2)
                    mask[s:e] = True
                    events.append(
                        {
                            "type": "sustained_drop",
                            "index": int(i),
                            "duration_days": run_len,
                            "total_drop_m": float(np.nansum(dh[i:j])),
                        }
                    )
                i = j
            else:
                i += 1

        return mask, events

    def _method_rolling_baseline(self, wl: np.ndarray, dry: np.ndarray, sig: float) -> np.ndarray:
        """Detect pumping via rolling-baseline deviation on dry days.

        This is NOT the PELT algorithm (Killick et al., 2012).  It is a
        simpler heuristic that:
        1. Computes a 45-day centred rolling mean as baseline.
        2. Flags dry-day segments where water level deviates below
           baseline by more than ``baseline_penalty * 3 * sigma``.
        3. Additionally flags short (5-day) rapid-drop windows.

        The ``baseline_penalty`` parameter controls sensitivity; higher
        values require larger deviations to trigger detection.
        """
        n = len(wl)
        mask = np.zeros(n, dtype=bool)
        window = 45
        min_seg = 10

        wl_filled = self._interpolate_nan(wl)
        roll_mean = pd.Series(wl_filled).rolling(window, center=True, min_periods=5).mean().values
        deviation = wl_filled - roll_mean

        th_level = -self.baseline_penalty * sig * 3
        level_drop = (deviation < th_level) & dry

        i = 0
        while i < n:
            if level_drop[i]:
                j = i
                while j < n and level_drop[j]:
                    j += 1
                if (j - i) >= min_seg:
                    mask[max(0, i - 2):min(n, j + 3)] = True
                i = j
            else:
                i += 1

        for start in range(0, n - 5):
            segment = wl_filled[start:start + 5]
            drop = segment[0] - np.nanmin(segment)
            if drop > 3 * sig and np.all(dry[start:start + 5]):
                mask[max(0, start - 1):min(n, start + 7)] = True

        return mask

    def _method_fourier(
        self,
        wl: np.ndarray,
        sig: float,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Detect periodic pumping-like signatures from detrended water levels."""
        n = len(wl)
        mask = np.zeros(n, dtype=bool)
        dominant_period = None

        wl_filled = self._interpolate_nan(wl)
        trend = pd.Series(wl_filled).rolling(30, center=True, min_periods=5).mean().values
        detrended = wl_filled - np.where(np.isnan(trend), wl_filled, trend)

        if len(detrended) < 30:
            return mask, dominant_period

        fft_vals = np.fft.rfft(detrended)
        freqs = np.fft.rfftfreq(n, d=1.0)
        power = np.abs(fft_vals) ** 2
        total_power = power.sum()

        if total_power < 1e-10:
            return mask, dominant_period

        target_periods = [7, 10, 14, 21, 30]
        for period in target_periods:
            target_freq = 1.0 / period
            freq_tol = 0.5 / period
            idx = np.where(
                (freqs > target_freq - freq_tol)
                & (freqs < target_freq + freq_tol)
                & (freqs > 0)
            )[0]
            if len(idx) == 0:
                continue

            period_power = power[idx].sum()
            if period_power / total_power > self.fourier_threshold:
                fft_copy = np.zeros_like(fft_vals)
                fft_copy[idx] = fft_vals[idx]
                component = np.fft.irfft(fft_copy, n=n)
                threshold = -1.5 * sig
                mask |= component < threshold
                dominant_period = float(period)

        return mask, dominant_period

    def _apply_known_pump(
        self,
        dates: pd.DatetimeIndex,
        known_pump_times: List[Tuple],
        n: int,
    ) -> np.ndarray:
        """Force known pumping intervals into the final mask."""
        mask = np.zeros(n, dtype=bool)
        for start, end in known_pump_times:
            s = pd.Timestamp(start)
            e = pd.Timestamp(end)
            idx = (dates >= s) & (dates <= e)
            mask[idx] = True
        return mask

    def _apply_buffer(self, mask: np.ndarray, buffer: int, n: int) -> np.ndarray:
        """Expand detected pumping intervals by a small temporal buffer."""
        if buffer <= 0:
            return mask
        out = mask.copy()
        for i in np.where(mask)[0]:
            s = max(0, i - buffer)
            e = min(n, i + buffer + 1)
            out[s:e] = True
        return out

    def _interpolate_nan(self, arr: np.ndarray) -> np.ndarray:
        """Fill missing values so rolling and spectral methods can run."""
        s = pd.Series(arr)
        return (
            s.interpolate(method="linear", limit_direction="both")
            .bfill()
            .ffill()
            .values
        )

    def _describe_events(
        self,
        dates: pd.DatetimeIndex,
        wl: np.ndarray,
        pump_mask: np.ndarray,
    ) -> List[dict]:
        """Summarize contiguous pumping intervals for downstream reporting."""
        events = []
        in_event = False
        start_i = 0

        for i, flag in enumerate(pump_mask):
            if flag and not in_event:
                in_event = True
                start_i = i
            elif not flag and in_event:
                in_event = False
                events.append(
                    {
                        "start_date": str(pd.Timestamp(dates[start_i]).date()),
                        "end_date": str(pd.Timestamp(dates[i - 1]).date()),
                        "duration_days": i - start_i,
                        "wl_drop_m": float(np.nanmax(wl[start_i:i]) - np.nanmin(wl[start_i:i]))
                        if np.any(~np.isnan(wl[start_i:i]))
                        else 0.0,
                    }
                )

        if in_event:
            n = len(pump_mask)
            events.append(
                {
                    "start_date": str(pd.Timestamp(dates[start_i]).date()),
                    "end_date": str(pd.Timestamp(dates[-1]).date()),
                    "duration_days": n - start_i,
                    "wl_drop_m": 0.0,
                }
            )

        return events
