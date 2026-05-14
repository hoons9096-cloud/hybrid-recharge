# Preprocessing Notes

This project contains two pumping-focused preprocessing modules:

- `pump_preprocess/preprocess/detector.py`
- `pump_preprocess/preprocess/corrector.py`

## Detector

`PumpingDetector` combines multiple heuristics:

- `sigma`: flags abrupt dry-day drops and sustained declines
- `pelt`: flags level-shift style anomalies over a rolling baseline
- `fourier`: flags periodic pumping-like signatures after detrending

The detector returns:

- a boolean pumping mask
- confidence values
- per-method masks
- event diagnostics

## Corrector

`WaterLevelCorrector` fills detected pumping segments using:

- `spline_fill` for short gaps
- `recession_fill` for medium gaps
- `baseline_shift` for longer contaminated intervals

The corrector returns:

- corrected water levels
- a filled-mask
- the dominant strategy used
- a recession coefficient estimate
- lightweight diagnostics

## Current cleanup status

Recent changes focused on deployability:

- shared file validation through `data_loader.py`
- fail-fast validation in `app_v30.py`
- baseline regression tests in `tests/`
- clarified preprocessing package entrypoint

## Remaining cleanup

Some historical comments and docstrings still contain broken text encoding.
Those sections should be cleaned in a dedicated pass, ideally by replacing the
affected files as a whole after confirming behavior with tests.
