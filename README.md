# Hybrid Recharge — Model-Discrepancy-Aware Water Table Fluctuation Framework

Companion code for

> **Choi, J. (2026).** *Bounding model discrepancy in water-table-fluctuation
> recharge estimates: a reproducible decision framework.*
> Submitted to *Environmental Modelling & Software*.

This repository implements a reproducible framework for spatial groundwater
recharge estimation and model-discrepancy reporting from water-level records.
It couples four methodological components:

1. **Soil-weighted upscaling** of WTF point estimates via national
   HSG-fraction polygons.
2. **Hierarchical Bayesian specific-yield inference** used as a diagnostic of
   parameter-data tension rather than the preferred recharge estimator.
3. **Learned discrepancy correction** with an attenuation axis α ∈ [0, 1],
   accompanied by explicit in-distribution and out-of-distribution tests.
4. **UZF and mixed-form Richards process references** that provide an
   independent model-structure range for reporting the field application.

The Yeongcheon, South Korea case study is an operational demonstration. The
main scientific claims are based on the fully reproducible synthetic benchmark.

## Reproducibility freeze for the submitted manuscript

The submission uses `bias_model.json` as the single source of truth for the
frozen eight-replicate cascade-trained discrepancy model:

- retained cells: 273,647
- five-fold CV R²: 0.6085
- mean multiplicative bias: −35.7% before correction, +1.3% after correction
- recharge RMSE: 285.8 to 153.9 mm yr⁻¹

The exact repository revision cited by the manuscript is recorded in the
Software and Data Availability section.

## Quick start

```bash
git clone https://github.com/hoons9096-cloud/hybrid-recharge.git
cd hybrid-recharge
pip install -r requirements.txt

python -m evaluation.benchmark_matrix \
    --scenarios S1 S2 S3 S4 S5 \
    --methods Lumped Soil-weighted Bias-corrected Hierarchical EnKF \
    --truth alpha cascade \
    --n_days 730 \
    --output benchmark_results.csv

python -m bias_correction --n_rep 8 --truth cascade --output bias_model
python -m unittest discover tests
```

Total runtime for the full synthetic benchmark and discrepancy-model fit on a
2023 Apple M2 (16 GB RAM) is under 18 minutes.

## Interactive UI

```bash
streamlit run app_v30_en.py
```

## Repository layout

| Path | Purpose |
|---|---|
| `app_v30.py`, `app_v30_en.py` | Streamlit interfaces |
| `bayes_hierarchical.py`, `bayes_sy.py` | Hierarchical Bayesian inference |
| `bias_correction.py` | Learned discrepancy regression and α-axis |
| `core_sim_v27.py` | Cascade vadose surrogate |
| `enkf_spatial.py` | Ensemble Kalman Filter benchmark |
| `fao56_swb.py` | FAO-56 soil water balance |
| `shp_soil_mapper.py` | HSG polygon-to-well mapping |
| `watershed_aggregator.py` | Soil-weighted aggregation |
| `evaluation/` | Benchmark and process-reference scripts |
| `synthetic/` | Synthetic domain and data generators |
| `pump_preprocess/` | Pump-detection preprocessing |
| `tests/` | Automated tests |

## Data sources used in the field demonstration

Raw field observations are not redistributed. They remain subject to the
source agencies' access terms.

| Dataset | Source | Access |
|---|---|---|
| Groundwater levels (KRC network) | Korea Rural Community Corporation | https://www.ekr.or.kr |
| Groundwater levels (national network) | GIMS | https://www.gims.go.kr |
| Climate forcing | Korea Meteorological Administration ASOS station 136 | https://apihub.kma.go.kr |
| National Detailed Soil Map | Rural Development Administration | https://soil.rda.go.kr |

The synthetic benchmark is reproducible from this repository alone and
requires no external data.

## License

MIT — see `LICENSE`.

## Contact

Junghoon Choi, GEOINNOVATION Co., Ltd., Daegu, Republic of Korea

ORCID: 0009-0002-0509-8089
