"""Tab 10 (English) — Watershed-scale recharge with bias-aware framework.

Mirrors ui/tab_watershed_report.py 1:1 with all UI text in English.
Includes Phase 5 (bias-aware) decision chart and multi-proxy envelope.
"""
from __future__ import annotations

import os
import traceback
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

try:
    from ui import TabContext
except ImportError:
    from dataclasses import dataclass
    @dataclass
    class TabContext:
        pass


def render(tab, ctx):
    with tab:
        st.markdown("### 🗾 Watershed recharge report (Lumped vs Soil-weighted)")
        st.caption(
            "Spatial upscaling using HSG (Hydrologic Soil Group) area "
            "fractions from a national digital soil map. "
            "Compares Lumped (baseline) vs Soil-weighted (proposed)."
        )

        try:
            import wells_registry as wr
            from wells_registry import WELLS, WATERSHEDS
            from watershed_aggregator import estimate_watershed
            from shp_soil_mapper import SHP_PATH_DEFAULT
        except Exception as e:
            st.error(f"Module import failed: {e}")
            return

        if not os.path.exists(SHP_PATH_DEFAULT):
            st.warning(
                f"⚠️ Soil shapefile not found: `{SHP_PATH_DEFAULT}`\n\n"
                "This tab requires the national digital soil map."
            )
            return

        _render_registry_manager_en(wr)
        WELLS = wr.WELLS
        WATERSHEDS = wr.WATERSHEDS
        if not WATERSHEDS:
            st.error("No wells registered. Use the registry manager above to add some.")
            return

        # ── Analysis mode ──
        try:
            from well_results_store import list_stored
            stored_list = list_stored()
        except Exception:
            stored_list = []
        stored_names = {s.well_name for s in stored_list}

        m1, m2 = st.columns([2, 3])
        with m1:
            mode = st.radio(
                "Analysis mode",
                options=["⚡ Use saved results (recommended)",
                         "🔄 Live recomputation"],
                index=0 if stored_list else 1,
                help=(
                    "Saved: reuse Tab 1 results (preserves per-well tuning).\n"
                    "Live: recompute from scratch with default parameters."
                ),
            )
            use_cached = mode.startswith("⚡")
        with m2:
            if stored_list:
                st.success(
                    f"📂 Currently saved wells: {len(stored_list)} — "
                    + ", ".join(sorted(stored_names))
                )
            else:
                st.info("ℹ️ No saved wells. Run Tab 1 → 💾 Save first.")

        col1, col2 = st.columns([2, 1])
        with col1:
            ws_choice = st.selectbox(
                "Select watershed",
                options=list(WATERSHEDS.keys()),
                index=0,
                help="Defined in wells_registry.WATERSHEDS",
            )
        with col2:
            buffer_km = st.number_input(
                "Well buffer (km)",
                min_value=0.5, max_value=10.0, value=2.0, step=0.5,
                help="Buffer radius around each well used to derive HSG fractions.",
            )

        well_names = WATERSHEDS[ws_choice]
        st.markdown(f"**Watershed `{ws_choice}` wells**: " + ", ".join(well_names))

        cwd = os.getcwd()
        file_paths: Dict[str, str] = {}
        if use_cached:
            cached_in_ws = [n for n in well_names if n in stored_names]
            missing = [n for n in well_names if n not in stored_names]
            if missing:
                st.warning(
                    f"⚠️ Wells without saved results: {missing} — "
                    "run Tab 1 → 💾 Save for these wells. Analysis still proceeds."
                )
            if not cached_in_ws:
                st.error("No saved wells in this watershed.")
                return
        else:
            missing = []
            for n in well_names:
                for cand in [f"{n}.txt", os.path.join(cwd, f"{n}.txt")]:
                    if os.path.exists(cand):
                        file_paths[n] = cand; break
                else:
                    missing.append(n)
            if missing:
                st.warning(f".txt files missing: {missing} — those wells skipped.")
            if not file_paths:
                st.error("No usable .txt files.")
                return

        if not st.button("🚀 Run watershed analysis", type="primary"):
            st.info("Click the button above to start analysis.")
            return

        spinner_msg = (
            f"{ws_choice} — loading saved results…"
            if use_cached else f"{ws_choice} — live recomputation…"
        )
        with st.spinner(spinner_msg):
            try:
                r = estimate_watershed(
                    ws_choice, file_paths=file_paths,
                    run_fao=False, buffer_km=buffer_km,
                    use_cached=use_cached,
                )
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.code(traceback.format_exc())
                return

        st.success(f"✅ {len(r.wells)} wells analysed")

        # 1. Soil distribution
        st.markdown("#### 1. Watershed soil distribution (HSG fractions)")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            fr_df = pd.DataFrame({
                "HSG": list(r.profile.hsg_fractions.keys()),
                "Area fraction": [f"{v*100:.1f}%" for v in r.profile.hsg_fractions.values()],
            })
            st.dataframe(fr_df, hide_index=True, use_container_width=True)
            st.metric("Total area (buffered)",
                      f"{r.profile.total_area_km2:.1f} km²")
            st.metric("Dominant HSG", r.profile.dominant_hsg)
        with col_b:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(4, 4))
                hsgs = list(r.profile.hsg_fractions.keys())
                vals = [r.profile.hsg_fractions[h]*100 for h in hsgs]
                colors = {"A": "#fde725", "B": "#5ec962",
                          "C": "#21918c", "D": "#3b528b"}
                ax.pie(vals, labels=[f"HSG {h}\n{v:.1f}%" for h, v in zip(hsgs, vals)],
                       colors=[colors.get(h, "gray") for h in hsgs],
                       startangle=90)
                ax.set_title(f"{ws_choice} — HSG distribution")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.caption(f"Pie chart failed: {e}")

        # 2. Per-well results
        st.markdown("#### 2. Per-well recharge")
        bayes_pcts = []
        well_rows = []
        for w in r.wells:
            row = {
                "Well": w.well.name,
                "Aquifer": w.well.aquifer,
                "HSG": w.soil.hydro_type,
                "sn": w.sn_used,
                "WTF (%)": f"{w.wtf_pct:.2f}" if w.wtf_pct is not None else "-",
                "WTF (mm/yr)": f"{w.wtf_mm:.0f}" if w.wtf_mm is not None else "-",
            }
            if use_cached:
                try:
                    from well_results_store import load as load_stored
                    s = load_stored(w.well.name)
                except Exception:
                    s = None
                if s and s.bayes_rech_pct_post_mean is not None:
                    row["Bayes (%) [95% CI]"] = (
                        f"{s.bayes_rech_pct_post_mean:.2f} "
                        f"[{s.bayes_rech_pct_post_lo95:.2f}, "
                        f"{s.bayes_rech_pct_post_hi95:.2f}]"
                    )
                    bayes_pcts.append({
                        "well": w.well.name, "hsg": w.soil.hydro_type,
                        "rech_pct": s.bayes_rech_pct_post_mean,
                        "lo95": s.bayes_rech_pct_post_lo95,
                        "hi95": s.bayes_rech_pct_post_hi95,
                    })
                else:
                    row["Bayes (%) [95% CI]"] = "—"
            well_rows.append(row)
        st.dataframe(pd.DataFrame(well_rows), hide_index=True,
                     use_container_width=True)

        # 3. Watershed summary — Lumped vs Soil-weighted
        st.markdown("#### 3. Watershed summary — Lumped vs Soil-weighted")
        col_l, col_s, col_d = st.columns(3)
        with col_l:
            st.metric("Lumped (baseline)",
                      f"{r.lumped_wtf_pct:.2f} %" if r.lumped_wtf_pct else "-",
                      help="Arithmetic mean across wells. Ignores soil heterogeneity.")
        with col_s:
            st.metric("Soil-weighted (proposed)",
                      f"{r.soil_weighted_wtf_pct:.2f} %" if r.soil_weighted_wtf_pct else "-",
                      help="HSG area-weighted. Missing-HSG fallback to overall mean.")
        with col_d:
            if r.lumped_wtf_pct and r.soil_weighted_wtf_pct:
                delta = r.soil_weighted_wtf_pct - r.lumped_wtf_pct
                st.metric("Δ (proposed − baseline)", f"{delta:+.2f} pp",
                          help="Effect of soil heterogeneity on recharge estimate.")

        if r.P_annual_mm:
            st.caption(
                f"📌 Mean annual P (from data): **{r.P_annual_mm:.0f} mm/yr** — "
                "extrapolated from short record; treat as approximate."
            )

        # 4. Bias-Aware WTF (Phase 5)
        _render_bias_aware_section_en(r)

        # 5. Hierarchical Bayesian (Phase 3) for cached mode
        if use_cached and bayes_pcts:
            _render_hierarchical_bayesian_en(
                ws_choice, [w.well.name for w in r.wells], r.profile,
            )

        # CSV download
        st.markdown("#### Export")
        rows = []
        for w in r.wells:
            rows.append({
                "watershed": ws_choice,
                "well": w.well.name,
                "aquifer": w.well.aquifer,
                "HSG": w.soil.hydro_type,
                "soil_code": w.soil.soil_code,
                "sn_idx": w.sn_used,
                "wtf_pct": w.wtf_pct,
                "wtf_mm": w.wtf_mm,
                "P_annual_mm": w.P_annual_mm,
            })
        rows.append({"watershed": ws_choice, "well": "<LUMPED>",
                      "wtf_pct": r.lumped_wtf_pct})
        rows.append({"watershed": ws_choice, "well": "<SOIL_WEIGHTED>",
                      "wtf_pct": r.soil_weighted_wtf_pct})
        csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv,
                           file_name=f"watershed_{ws_choice}_recharge.csv",
                           mime="text/csv")


# ---------------------------------------------------------------------------
# Phase 5 — Bias-aware WTF (English)
# ---------------------------------------------------------------------------
def _render_bias_aware_section_en(r):
    import os, json
    st.markdown("#### 4. 🎯 Bias-Aware WTF (Phase 5 — learned bias correction)")
    st.caption(
        "Corrects the structural bias of the WTF identity using a regression "
        "trained on the cascade vadose truth. Tune α to control "
        "conservatism (0=no correction, 0.3=recommended, 1=full)."
    )

    bm_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bias_model.json",
    )
    if not os.path.exists(bm_path):
        st.warning(
            "⚠️ `bias_model.json` not found — run "
            "`python -m bias_correction --n_rep 8 --output bias_model` first."
        )
        return
    with open(bm_path) as f:
        bm = json.load(f)
    coefs = np.array(bm["coefs"])

    col_a, col_b = st.columns([2, 1])
    with col_a:
        alpha = st.slider(
            "Conservatism α", min_value=0.0, max_value=1.0,
            value=0.3, step=0.05,
            help="0 = no correction, 0.3 = recommended (Korean lit. range), "
                 "1 = full cascade-strength",
        )
    with col_b:
        ET_over_P = st.number_input(
            "ET/P (regional climate)", min_value=0.1, max_value=1.0,
            value=0.5, step=0.05,
            help="FAO-56-based Korean monsoon mean ≈ 0.5",
        )

    try:
        from soil_db import SOIL_DB
        from watershed_aggregator import HSG_TO_SN_ALLUVIAL
    except Exception as e:
        st.error(f"Module import failed: {e}")
        return

    obs_sigma = 0.02
    betas, weights = [], []
    for hsg, frac in r.profile.hsg_fractions.items():
        if frac < 0.01:
            continue
        sn = HSG_TO_SN_ALLUVIAL.get(hsg, 12)
        sr = SOIL_DB[sn]
        x = np.array([1.0, sr.sy_lit, np.log(sr.tau), ET_over_P, obs_sigma])
        betas.append(float(np.dot(coefs, x)))
        weights.append(frac)
    if not betas:
        st.warning("No HSG distribution → correction unavailable")
        return
    weights = np.array(weights) / sum(weights)
    beta_hat = float(np.sum(np.array(betas) * weights))

    sw_pct = r.soil_weighted_wtf_pct or 0.0
    bc_results = {}
    for a_label, a_val in [("α=0 (raw)", 0.0), ("α=0.3 (default)", 0.3),
                            ("α=0.5", 0.5), ("α=1.0 (full)", 1.0),
                            (f"α={alpha:.2f} (selected)", alpha)]:
        denom = max(1.0 + a_val * beta_hat, 0.05)
        bc_results[a_label] = sw_pct / denom

    st.markdown("**At the selected α:**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Soil-weighted (α=0)", f"{sw_pct:.2f} %")
    with c2:
        delta = bc_results[f"α={alpha:.2f} (selected)"] - sw_pct
        st.metric(f"Bias-corrected (α={alpha:.2f})",
                  f"{bc_results[f'α={alpha:.2f} (selected)']:.2f} %",
                  delta=f"{delta:+.2f} pp vs raw")
    with c3:
        st.metric("β̂ (correction factor)", f"{beta_hat:+.3f}",
                  help="Negative β̂ = WTF under-predicts truth → correction inflates.")

    df_alpha = pd.DataFrame({
        "α": list(bc_results.keys()),
        "Recharge (%)": [f"{v:.2f}" for v in bc_results.values()],
    })
    st.dataframe(df_alpha, hide_index=True, use_container_width=False)

    st.markdown("---")
    st.markdown("**🛡️ Multi-proxy consistency check (synthetic-independent)**")
    try:
        from evaluation.proxy_validation import proxy_envelope
        env = proxy_envelope(P_annual_mm=1100.0)
    except Exception as e:
        st.caption(f"⚠️ Proxy module load failed: {e}")
        return

    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_hrect(y0=env.envelope_lo, y1=env.envelope_hi,
                      fillcolor="#10B981", opacity=0.10, line_width=0,
                      annotation_text="Multi-proxy envelope",
                      annotation_position="top right")
        fig.add_hrect(y0=env.molit_lo, y1=env.molit_hi,
                      fillcolor="#F59E0B", opacity=0.18, line_width=0)
        fig.add_hrect(y0=env.cmb_lo, y1=env.cmb_hi,
                      fillcolor="#7C3AED", opacity=0.13, line_width=0)
        fig.add_hrect(y0=env.bfi_lo, y1=env.bfi_hi,
                      fillcolor="#0891B2", opacity=0.08, line_width=0)

        alphas_grid = np.linspace(0, 1, 51)
        rech_curve = [sw_pct / max(1.0 + a * beta_hat, 0.05) for a in alphas_grid]
        fig.add_trace(go.Scatter(
            x=alphas_grid, y=rech_curve, mode="lines",
            name=f"{r.watershed} recharge",
            line=dict(color="#DC2626", width=3),
        ))
        for a_mark, sym in zip([0.0, 0.3, 1.0], ["circle", "square", "star"]):
            R_mark = sw_pct / max(1.0 + a_mark * beta_hat, 0.05)
            fig.add_trace(go.Scatter(
                x=[a_mark], y=[R_mark], mode="markers+text",
                marker=dict(symbol=sym, size=14, color="#DC2626",
                            line=dict(color="black", width=1)),
                text=[f"  {R_mark:.1f}%"], textposition="middle right",
                showlegend=False,
            ))
        R_user = sw_pct / max(1.0 + alpha * beta_hat, 0.05)
        fig.add_trace(go.Scatter(
            x=[alpha], y=[R_user], mode="markers",
            marker=dict(symbol="diamond", size=18, color="#F59E0B",
                        line=dict(color="black", width=2)),
            name=f"Selected α={alpha:.2f}",
        ))
        fig.update_layout(
            title=f"α-spectrum decision chart — {r.watershed}",
            xaxis_title="Conservatism α",
            yaxis_title="Recharge ratio (% of P)",
            xaxis=dict(range=[-0.05, 1.05]),
            yaxis=dict(range=[0, max(50, max(rech_curve) * 1.2)]),
            height=420, margin=dict(l=50, r=20, t=50, b=50),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

        st.caption(
            f"**Envelope check**: BFI {env.bfi_lo:.0f}–{env.bfi_hi:.0f}%, "
            f"CMB {env.cmb_lo:.0f}–{env.cmb_hi:.0f}%, "
            f"MOLIT {env.molit_lo:.0f}–{env.molit_hi:.0f}%. "
            "If selected α lies inside the envelope: ✅ operationally defensible."
        )
        if env.in_envelope(R_user):
            st.success(
                f"✅ At α={alpha:.2f} ({R_user:.2f}%) the estimate is "
                "inside the multi-proxy envelope."
            )
        else:
            st.warning(
                f"⚠️ At α={alpha:.2f} ({R_user:.2f}%) the estimate is "
                "outside the envelope — try a lower α."
            )
    except Exception as e:
        st.caption(f"Plotly chart failed: {e}")


# ---------------------------------------------------------------------------
# Hierarchical Bayesian (English)
# ---------------------------------------------------------------------------
def _render_hierarchical_bayesian_en(ws_choice, well_names, profile):
    st.markdown("#### 5. 🔮 Hierarchical Bayesian (Phase 3)")
    st.caption(
        "3-level hierarchy (watershed → HSG → well) via affine-invariant "
        "ensemble MCMC (emcee). Returns posterior Sy + watershed-mean "
        "recharge with 95% CI."
    )
    if not st.button("🔮 Run hierarchical Bayesian (tens of seconds)",
                     key=f"run_h_bayes_en_{ws_choice}"):
        st.info("Click to compute the hierarchical posterior.")
        return
    try:
        from bayes_hierarchical import fit_from_stored
        with st.spinner("emcee MCMC sampling…"):
            res = fit_from_stored(ws_choice, well_names,
                                   hsg_fractions=profile.hsg_fractions,
                                   n_walkers=24, n_steps=2000, burn_in=500)
    except Exception as e:
        st.error(f"Hierarchical inference failed: {e}")
        return

    st.caption(
        f"📊 N samples={res.n_samples}, "
        f"acceptance={res.mean_acceptance_rate:.2f}, "
        f"converged={'✅' if res.converged else '⚠️'}"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.metric("μ_watershed (Sy)", f"{res.mu_watershed_mean:.3f}")
        st.caption(f"95% CI [{res.mu_watershed_lo95:.3f}, {res.mu_watershed_hi95:.3f}]")
    with col2:
        if np.isfinite(res.rech_pct_watershed_mean):
            st.metric("Watershed recharge (Hierarchical)",
                       f"{res.rech_pct_watershed_mean:.2f} %")
            st.caption(
                f"95% CI [{res.rech_pct_watershed_lo95:.2f}, "
                f"{res.rech_pct_watershed_hi95:.2f}] %"
            )

    st.markdown("**HSG-level posterior μ**")
    rows = []
    for h, (m, lo, hi) in res.mu_hsg_summary.items():
        rows.append({"HSG": h, "Sy mean": f"{m:.3f}",
                      "95% CI": f"[{lo:.3f}, {hi:.3f}]",
                      "Area frac": f"{profile.hsg_fractions.get(h, 0)*100:.1f}%"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("**Per-well posterior Sy**")
    rows = []
    for i, n in enumerate(res.well_names):
        rows.append({"Well": n, "HSG": res.hsgs[i],
                      "Aquifer": res.aquifers[i],
                      "Sy posterior": f"{res.sy_well_mean[i]:.3f}",
                      "95% CI": f"[{res.sy_well_lo95[i]:.3f}, {res.sy_well_hi95[i]:.3f}]"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Registry CRUD (English)
# ---------------------------------------------------------------------------
def _render_registry_manager_en(wr):
    with st.expander("🛠️ Well / watershed manager (add · edit · delete)",
                     expanded=False):
        WELLS = wr.WELLS
        WATERSHEDS = wr.WATERSHEDS

        if WELLS:
            rows = []
            for name, w in WELLS.items():
                rows.append({"Well": name, "Watershed": w.watershed,
                              "lat": w.lat, "lon": w.lon,
                              "Aquifer": w.aquifer, "ASOS": w.nearest_kma})
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)
        else:
            st.info("No wells registered yet.")

        tab_add, tab_edit, tab_ws = st.tabs([
            "➕ Add well", "✏️ Edit / delete", "🗂️ Rename watershed",
        ])

        with tab_add:
            with st.form("add_well_en", clear_on_submit=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    new_name = st.text_input("Well name",
                                              placeholder="e.g. Daejeon-Yuseong1")
                    new_ws = st.text_input("Watershed",
                                            placeholder="e.g. Gabcheon")
                with c2:
                    new_lat = st.number_input("Latitude (WGS84)",
                                                min_value=33.0, max_value=39.0,
                                                value=36.35, step=0.0001,
                                                format="%.4f")
                    new_lon = st.number_input("Longitude (WGS84)",
                                                min_value=124.0, max_value=132.0,
                                                value=127.37, step=0.0001,
                                                format="%.4f")
                with c3:
                    new_aq = st.selectbox("Aquifer",
                                            ["alluvial", "bedrock"], index=1)
                    new_kma = st.number_input("KMA ASOS station ID",
                                                min_value=100, max_value=300,
                                                value=133, step=1)
                if st.form_submit_button("➕ Register", type="primary"):
                    try:
                        info = wr.add_well(new_name, new_lat, new_lon, new_ws,
                                            aquifer=new_aq,
                                            nearest_kma=int(new_kma))
                        st.success(f"✅ Added: {info.name} ({info.watershed})")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

        with tab_edit:
            if not WELLS:
                st.info("No wells to edit")
            else:
                tgt = st.selectbox("Select well", options=list(WELLS.keys()),
                                    key="edit_target_en")
                w = WELLS[tgt]
                with st.form(f"edit_en_{tgt}"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        e_ws = st.text_input("Watershed", value=w.watershed)
                    with c2:
                        e_lat = st.number_input("Latitude", value=float(w.lat),
                                                 step=0.0001, format="%.4f")
                        e_lon = st.number_input("Longitude", value=float(w.lon),
                                                 step=0.0001, format="%.4f")
                    with c3:
                        e_aq = st.selectbox("Aquifer",
                                              ["alluvial", "bedrock"],
                                              index=0 if w.aquifer == "alluvial" else 1)
                        e_kma = st.number_input("ASOS ID",
                                                 value=int(w.nearest_kma), step=1)
                    bsave, bdel = st.columns(2)
                    do_save = bsave.form_submit_button("💾 Save changes")
                    do_del = bdel.form_submit_button("🗑️ Delete", type="primary")
                    if do_save:
                        try:
                            wr.update_well(tgt, lat=e_lat, lon=e_lon,
                                            watershed=e_ws, aquifer=e_aq,
                                            nearest_kma=int(e_kma))
                            st.success(f"✅ Updated: {tgt}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {e}")
                    if do_del:
                        if wr.remove_well(tgt):
                            st.success(f"🗑️ Deleted: {tgt}")
                            st.rerun()

        with tab_ws:
            if not WATERSHEDS:
                st.info("No watersheds")
            else:
                with st.form("rename_ws_en"):
                    old = st.selectbox("Existing watershed name",
                                        options=list(WATERSHEDS.keys()))
                    new = st.text_input("New watershed name")
                    if st.form_submit_button("🔄 Rename"):
                        if new.strip():
                            n = wr.rename_watershed(old, new.strip())
                            st.success(f"✅ {n} wells renamed: {old} → {new}")
                            st.rerun()
