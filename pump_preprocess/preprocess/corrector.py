"""Water-level correction strategies for pumping-affected segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit


@dataclass
class CorrectionResult:
    """Container for correction outputs."""

    corrected_wl: np.ndarray
    filled_mask: np.ndarray
    strategy_used: str
    recession_k: Optional[float]
    rmse_check: Optional[float]
    diagnostics: dict


class WaterLevelCorrector:
    """Fill pumping-contaminated segments with simple, interpretable strategies."""

    def __init__(
        self,
        strategy: str = "auto",
        recession_window: int = 30,
        min_recession_days: int = 5,
        smoothing_sigma: float = 2.0,
        max_gap_spline: int = 7,
    ):
        self.strategy = strategy
        self.recession_window = recession_window
        self.min_recession_days = min_recession_days
        self.smoothing_sigma = smoothing_sigma
        self.max_gap_spline = max_gap_spline

    def correct(
        self,
        dates: pd.DatetimeIndex,
        water_level: np.ndarray,
        pump_mask: np.ndarray,
        rainfall: Optional[np.ndarray] = None,
    ) -> CorrectionResult:
        """Correct water levels inside detected pumping intervals."""
        del dates  # Reserved for future date-aware strategies.

        wl = np.array(water_level, dtype=float)
        mask = np.array(pump_mask, dtype=bool)
        n = len(wl)

        if not mask.any():
            return CorrectionResult(
                corrected_wl=wl.copy(),
                filled_mask=np.zeros(n, dtype=bool),
                strategy_used="none",
                recession_k=None,
                rmse_check=None,
                diagnostics={"message": "No pumping detected, no correction needed."},
            )

        k_recession = self._estimate_recession_k(wl, mask, rainfall)

        corrected = wl.copy()
        filled = np.zeros(n, dtype=bool)
        strategy_log = []
        segments = self._get_segments(mask, n)

        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            strategy = self._choose_strategy(seg_len)
            strategy_log.append(strategy)

            if strategy == "spline_fill":
                filled_seg = self._spline_fill(corrected, seg_start, seg_end, n)
            elif strategy == "recession_fill":
                filled_seg = self._recession_fill(corrected, seg_start, seg_end, k_recession, n)
            else:
                filled_seg = self._baseline_shift(corrected, seg_start, seg_end, n)

            corrected[seg_start:seg_end] = filled_seg
            filled[seg_start:seg_end] = True

        if self.smoothing_sigma > 0:
            corrected = self._smooth_boundaries(corrected, filled, self.smoothing_sigma)

        rmse_check = self._check_rmse(wl, corrected, ~mask)
        main_strategy = max(set(strategy_log), key=strategy_log.count) if strategy_log else "none"

        return CorrectionResult(
            corrected_wl=corrected,
            filled_mask=filled,
            strategy_used=main_strategy,
            recession_k=k_recession,
            rmse_check=rmse_check,
            diagnostics={
                "segments_corrected": len(segments),
                "strategies_used": strategy_log,
                "recession_k": k_recession,
                "total_filled_days": int(filled.sum()),
            },
        )

    def _choose_strategy(self, seg_len: int) -> str:
        """Choose a filling strategy based on contaminated segment length."""
        if self.strategy != "auto":
            return self.strategy
        if seg_len <= self.max_gap_spline:
            return "spline_fill"
        if seg_len <= 30:
            return "recession_fill"
        return "baseline_shift"

    def _recession_fill(
        self,
        wl: np.ndarray,
        seg_start: int,
        seg_end: int,
        k: float,
        n: int,
    ) -> np.ndarray:
        """Extend the pre-pumping recession curve through the contaminated gap."""
        pre_idx = seg_start - 1
        while pre_idx >= 0 and np.isnan(wl[pre_idx]):
            pre_idx -= 1

        if pre_idx < 0:
            return self._spline_fill(wl, seg_start, seg_end, n)

        h0 = wl[pre_idx]
        h_eq = np.nanpercentile(wl, 5)
        seg_len = seg_end - seg_start

        t = np.arange(1, seg_len + 1)
        filled = h_eq + (h0 - h_eq) * np.exp(k * t)

        post_idx = seg_end
        while post_idx < n and np.isnan(wl[post_idx]):
            post_idx += 1

        if post_idx < n:
            h_post = wl[post_idx]
            blend_len = min(3, seg_len)
            for i in range(blend_len):
                alpha = (i + 1) / (blend_len + 1)
                idx = seg_len - blend_len + i
                filled[idx] = (1 - alpha) * filled[idx] + alpha * h_post

        return filled

    def _spline_fill(
        self,
        wl: np.ndarray,
        seg_start: int,
        seg_end: int,
        n: int,
    ) -> np.ndarray:
        """Interpolate a short pumping interval from nearby valid points."""
        pre_valid = []
        for i in range(max(0, seg_start - 10), seg_start):
            if not np.isnan(wl[i]):
                pre_valid.append((i, wl[i]))
        pre_valid = pre_valid[-5:]

        post_valid = []
        for i in range(seg_end, min(n, seg_end + 10)):
            if not np.isnan(wl[i]):
                post_valid.append((i, wl[i]))
        post_valid = post_valid[:5]

        all_pts = pre_valid + post_valid
        if len(all_pts) < 2:
            return np.linspace(
                wl[max(0, seg_start - 1)] if seg_start > 0 else np.nanmean(wl),
                wl[min(n - 1, seg_end)] if seg_end < n else np.nanmean(wl),
                seg_end - seg_start,
            )

        xs = np.array([p[0] for p in all_pts])
        ys = np.array([p[1] for p in all_pts])

        if len(xs) >= 4:
            cs = CubicSpline(xs, ys, bc_type="natural")
            return cs(np.arange(seg_start, seg_end))

        return np.interp(np.arange(seg_start, seg_end), xs, ys)

    def _baseline_shift(
        self,
        wl: np.ndarray,
        seg_start: int,
        seg_end: int,
        n: int,
    ) -> np.ndarray:
        """Bridge long contaminated gaps with a smooth baseline between both sides."""
        pre_window = min(self.recession_window, seg_start)
        post_window = min(self.recession_window, n - seg_end)

        h_pre = (
            np.nanmedian(wl[max(0, seg_start - pre_window):seg_start])
            if pre_window > 0
            else np.nanmean(wl)
        )
        h_post = (
            np.nanmedian(wl[seg_end:min(n, seg_end + post_window)])
            if post_window > 0
            else np.nanmean(wl)
        )

        seg_len = seg_end - seg_start
        return np.linspace(h_pre, h_post, seg_len)

    def _estimate_recession_k(
        self,
        wl: np.ndarray,
        pump_mask: np.ndarray,
        rainfall: Optional[np.ndarray],
    ) -> float:
        """Estimate a recession coefficient from non-pumping dry-day declines."""
        valid = ~pump_mask & ~np.isnan(wl)
        if rainfall is not None:
            valid &= rainfall < 1.0

        wl_valid = wl[valid]
        if len(wl_valid) < self.min_recession_days:
            return -0.015

        dh = np.diff(wl_valid)
        neg_idx = np.where(dh < 0)[0]
        if len(neg_idx) < 3:
            return -0.015

        try:
            runs = []
            run = [neg_idx[0]]
            for i in range(1, len(neg_idx)):
                if neg_idx[i] == neg_idx[i - 1] + 1:
                    run.append(neg_idx[i])
                else:
                    if len(run) >= self.min_recession_days:
                        runs.append(run)
                    run = [neg_idx[i]]
            if len(run) >= self.min_recession_days:
                runs.append(run)

            if not runs:
                return -0.015

            longest = max(runs, key=len)
            t = np.arange(len(longest) + 1, dtype=float)
            h = wl_valid[longest[0]:longest[-1] + 2]
            if len(h) < 3:
                return -0.015

            h0 = h[0]
            h_min = np.nanmin(h)

            def exp_decay(t_val, k_val, c_val):
                return (h0 - c_val) * np.exp(k_val * t_val) + c_val

            popt, _ = curve_fit(
                exp_decay,
                t[:len(h)],
                h,
                p0=[-0.02, h_min],
                bounds=([-1.0, -np.inf], [-0.001, np.inf]),
                maxfev=2000,
            )
            k_fit = float(popt[0])
            return max(min(k_fit, -0.001), -0.5)
        except Exception:
            return -0.015

    def _get_segments(self, mask: np.ndarray, n: int):
        """Return contiguous pumping intervals as `(start, end)` pairs."""
        segments = []
        in_seg = False
        start = 0
        for i in range(n):
            if mask[i] and not in_seg:
                in_seg = True
                start = i
            elif not mask[i] and in_seg:
                in_seg = False
                segments.append((start, i))
        if in_seg:
            segments.append((start, n))
        return segments

    def _smooth_boundaries(
        self,
        wl: np.ndarray,
        filled: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """Blend the edges of filled segments to reduce sharp transitions."""
        out = wl.copy()
        boundaries = []
        for i in range(1, len(filled)):
            if filled[i] != filled[i - 1]:
                boundaries.append(i)

        half_w = max(1, int(sigma * 2))
        for b in boundaries:
            for j in range(max(0, b - half_w), min(len(wl), b + half_w)):
                dist = abs(j - b)
                w_orig = np.exp(-0.5 * (dist / sigma) ** 2)
                out[j] = w_orig * wl[j] + (1 - w_orig) * out[j]
        return out

    def _check_rmse(
        self,
        original: np.ndarray,
        corrected: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Optional[float]:
        """Measure how much non-pumping values changed after correction."""
        idx = valid_mask & ~np.isnan(original) & ~np.isnan(corrected)
        if idx.sum() < 5:
            return None
        diff = original[idx] - corrected[idx]
        return float(np.sqrt(np.mean(diff ** 2)))
