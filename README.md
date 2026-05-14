# Hybrid Recharge — Bias-Aware Water Table Fluctuation Framework

Companion code for

> **Choi, J. (2026).** *Bias-Aware Water Table Fluctuation: From Point
> Estimator to Decision Framework for Groundwater Recharge.*
> Submitted to *Journal of Hydrology*.

This repository implements a bias-aware framework for spatial groundwater
recharge estimation from water-level records. It couples four
methodological components, summarised in the paper:

1. **Soil-weighted upscaling** of WTF point estimates via national
   HSG-fraction polygons (Section 2.1).
2. **Hierarchical Bayesian S<sub>y</sub> inference** used as a
   *diagnostic* of the WTF identity's structural bias (Section 2.2;
   §3.3.1 of the paper).
3. **Learned bias-correction regression** with a conservatism axis
   α ∈ [0, 1] and a (α = 0, 0.3, 1) reporting triple (Section 4).
4. **Multi-proxy + UZF–Richards model-structure envelope** that bounds
   the field estimate independently of the cascade-class synthetic
   truth (Sections 4.6, 6.6).

The framework is designed to be applied to *any* combination of
water-level records, national soil map, and climate forcing — the
Yeongcheon (South Korea) case study reported in Sections 5–6 of the
paper is one specific deployment.

## Quick start

```bash
git clone https://github.com/hoons9096-cloud/hybrid-recharge.git
cd hybrid-recharge
pip install -r requirements.txt

# Run the synthetic benchmark (Tables 2, 2b, 5, A1 of the paper)
python -m evaluation.benchmark_matrix \
    --scenarios S1 S2 S3 S4 S5 \
    --methods Lumped Soil-weighted Bias-corrected Hierarchical EnKF \
    --truth alpha cascade \
    --n_days 730 \
    --output benchmark_results.csv

# Fit the bias-correction regression (Table 4 + cross-truth tests)
python -m bias_correction --n_rep 8 --truth cascade --output bias_model

# Run the unit-test suite (33 tests, ~17 s on Apple M2)
python -m unittest discover tests
```

Total runtime for the full synthetic benchmark + bias-correction fit on
a 2023 Apple M2 (16 GB RAM): **< 18 minutes**.

## Interactive UI

A Streamlit interface (`app_v30.py`) exposes the full per-well + watershed
analysis workflow:

```bash
streamlit run app_v30.py
```

## Repository layout

| Path | Purpose |
|---|---|
| `app_v30.py` | Streamlit UI (Korean) |
| `app_v30_en.py` | Streamlit UI (English) |
| `bayes_hierarchical.py`, `bayes_sy.py` | Hierarchical Bayesian S<sub>y</sub> inference |
| `bias_correction.py` | Learned bias-correction regression with α-spectrum |
| `core_sim_v27.py` | Cascade vadose model (synthetic truth) |
| `enkf_spatial.py` | Ensemble Kalman Filter benchmark |
| `fao56_swb.py` | FAO-56 soil water balance (independent ET<sub>a</sub>) |
| `kma_adapter.py` | Korea Meteorological Administration data adapter |
| `shp_soil_mapper.py` | National HSG soil map polygon-to-well mapping |
| `watershed_aggregator.py` | Soil-weighted watershed aggregation |
| `evaluation/` | Benchmark scripts, figures, UZF / Richards reference solvers |
| `methods/` | Method-level wrappers (Lumped, Soil-weighted, etc.) |
| `synthetic/` | Synthetic domain + data generators |
| `pump_preprocess/` | Pump-detection pre-processing |
| `tests/` | 33 unit tests covering cascade verification, bias correction, hierarchical Bayes, EnKF |
| `ui/`, `ui_en/` | Streamlit page modules |

## Data sources used in the paper

The Yeongcheon field application of the paper uses the following data
sources. Raw observation records are not redistributed in this
repository; readers wishing to reproduce the Yeongcheon results should
obtain them directly from the agencies below.

| Dataset | Source | Access |
|---|---|---|
| Groundwater levels (KRC network) | Korea Rural Community Corporation groundwater observation network | https://www.ekr.or.kr |
| Groundwater levels (national network) | National Groundwater Monitoring Network / GIMS | https://www.gims.go.kr |
| Climate forcing (precipitation, T<sub>min</sub>, T<sub>max</sub>) | Korea Meteorological Administration ASOS, station 136 (Yeongcheon) | https://apihub.kma.go.kr |
| National Detailed Soil Map (HSG, EPSG:5186) | Rural Development Administration of Korea | https://soil.rda.go.kr |
| MOLIT Korean Recharge Atlas (2016) | Ministry of Land, Infrastructure and Transport | publicly distributed |

The synthetic benchmark of the paper (Sections 3–4) is fully
reproducible from this repository alone, requiring no external data.

## Reproducibility checklist

- [x] All synthetic results in Tables 2–7, A1 of the paper are
      reproducible from the synthetic benchmark scripts.
- [x] All Yeongcheon-derived per-soil and watershed-mean numbers
      reported in Sections 5–6 can be reproduced by supplying the
      external data above and re-running the `evaluation/*_yeongcheon.py`
      scripts.
- [x] The UZF and HYDRUS-class Richards reference solvers used in
      Section 6.6 are independent of the cascade-class synthetic truth
      and can be re-run against any USDA-class soil and climate input
      (`evaluation/run_cascade_uzf_yeongcheon.py`,
      `evaluation/run_richards_yeongcheon.py`).

## Citation

If you use this framework in your work, please cite:

```bibtex
@article{Choi2026WTFKalman,
  author  = {Choi, Junghoon},
  title   = {Bias-Aware Water Table Fluctuation: From Point Estimator
             to Decision Framework for Groundwater Recharge},
  journal = {Journal of Hydrology},
  year    = {2026},
  note    = {Submitted}
}
```

## Contact

**Junghoon Choi, Ph.D.**
Founder & CEO, GEOINNOVATION Co., Ltd.
Daegu, Republic of Korea
ORCID: [0009-0002-0509-8089](https://orcid.org/0009-0002-0509-8089)

GEOINNOVATION Co., Ltd. is a hydrogeology consulting firm specialising
in groundwater impact assessment, dam-effect analysis, drought-response
planning, and basin-scale recharge mapping. The framework provided here
is the methodological backbone of the firm's recharge-estimation
services and is released as open source to permit independent
reproduction and adaptation beyond the firm's own consulting practice.

## License

MIT — see [LICENSE](LICENSE).
