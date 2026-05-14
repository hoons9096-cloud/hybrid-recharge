"""Tab 1 (English) — Single-well basic analysis.

English variant of ui/tab_base.py.  Mirrors the Korean version 1:1 but
with all user-facing text in English, suitable for international
reviewers and operational users.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ui import (
    C, SOIL_NAMES, TabContext,
    build_hybrid_radar, shade_pump_plotly,
)


def render(tab, ctx: TabContext):
    with tab:
        result = st.session_state.get("result_v27")
        if result is None:
            st.info("Sidebar → run **'Step 1: Basic analysis'** first.")
            return

        st.markdown("### 📊 Basic analysis (v27)")

        pump_idx = float(result.get("pump_contam_idx", 0))
        sy_eff = float(result.get("Sy_eff", 0))
        rr_v27 = float(result["recharge_ratio"])
        rr_corr = result.get("recharge_ratio_corrected")
        has_pump_corr = rr_corr is not None

        # ── Row 1: Primary metrics ──
        if has_pump_corr:
            cA, cB, cC, cC2, cD = st.columns(5)
        else:
            cA, cB, cC, cD = st.columns(4)
        cA.metric("RMSE", f"{float(result['rmse']):.4f} m")
        cB.metric("CC", f"{float(result['cc']):.4f}")
        cC.metric("Recharge (v27 WTF)", f"{rr_v27:.2f}%")
        if has_pump_corr:
            delta = float(rr_corr) - rr_v27
            cC2.metric("Recharge (corrected)", f"{float(rr_corr):.2f}%",
                       delta=f"{delta:+.2f}pp")
        cD.metric("Pump contamination idx", f"{pump_idx:.2f}")

        # ── Row 2: Hydrological metrics (Moriasi 2007) ──
        nse_val = result.get("nse")
        if nse_val is not None:
            cE, cF, cG, cH = st.columns(4)
            nse_f = float(nse_val)
            nse_c = "🟢" if nse_f >= 0.75 else ("🟡" if nse_f >= 0.5 else "🔴")
            cE.metric(f"NSE {nse_c}", f"{nse_f:.3f}")
            kge_f = float(result.get("kge") or 0.0)
            kge_c = "🟢" if kge_f >= 0.75 else ("🟡" if kge_f >= 0.5 else "🔴")
            cF.metric(f"KGE {kge_c}", f"{kge_f:.3f}")
            pb_f = float(result.get("pbias") or 0.0)
            pb_c = "🟢" if abs(pb_f) < 10 else ("🟡" if abs(pb_f) < 25 else "🔴")
            cG.metric(f"PBIAS {pb_c}", f"{pb_f:+.1f}%")
            cH.metric("Sy (effective)", f"{sy_eff:.4f}")

        if has_pump_corr:
            st.caption(
                "ℹ️ **v27 WTF**: raw water level | "
                "**Corrected**: same v27 re-run after pumping removal."
            )
        if pump_idx >= 0.45:
            st.error("🚫 High pumping contamination → run pumping pre-processing (Tab 2).")
        elif pump_idx >= 0.25:
            st.warning("⚠️ Suspected pumping influence → try pre-processing (Tab 2).")

        bw = result.get("boundary_warnings", [])
        if bw:
            for msg in bw:
                st.warning(f"⚠️ Optimisation warning: {msg}")
        else:
            st.success("✅ Pumping contamination low")

        _render_scan_results()
        _render_main_chart(result, ctx, rr_v27)
        _render_bayesian_sy(result, ctx)
        _render_save_for_watershed(result, ctx, has_pump_corr)
        _render_ai_opinion_en(result, ctx, rr_v27, rr_corr,
                              has_pump_corr, sy_eff)


# ---------------------------------------------------------------------------
# Scan / BMA section
# ---------------------------------------------------------------------------
def _render_scan_results():
    scan_df = st.session_state.get("scan_data")
    if scan_df is None:
        st.info(
            "ℹ️ Sidebar → run **'② Hybrid soil precision scan'** to display "
            "the soil-recommendation panel.\n\n"
            "💡 With Auto-Optimize off, dragging the k/z sliders shows how the "
            "recharge curve responds — useful for parameter-sensitivity intuition."
        )
        return
    with st.expander("🛡️ Soil recommendation (Hybrid scan)", expanded=True):
        best_row = st.session_state.get("best_soil")
        if best_row is None:
            st.info("Scan completed but no recommendation found.")
            return
        conf = st.session_state.get("best_soil_conf", "MEDIUM")
        tentative = bool(st.session_state.get("best_soil_tentative", False))
        col_a, col_b = st.columns([1, 2])
        with col_a:
            label = best_row["Soil"] + (" (tentative)" if tentative else "")
            st.success(f"🏆 Best soil: **{label}**")
            st.metric("TOPSIS score",
                      f"{best_row.get('TopsisScore', best_row['HybridScore']):.1f}")
            st.write(f"Confidence: **{conf}**")
            st.write(f"Pump idx: `{best_row.get('PumpIdx', 0):.2f}`")
        with col_b:
            cols = ["Soil", "TopsisScore", "StressScore", "SyScore",
                    "SlopeErr", "PumpIdx", "EvalN", "RecoFlag"]
            avail = [c for c in cols if c in scan_df.columns]
            disp = scan_df[avail].head(5).copy()
            st.dataframe(disp, hide_index=True, use_container_width=True)

        # ── BMA (Bayesian Model Averaging) ──
        bma_res = st.session_state.get("bma_result")
        if bma_res is not None:
            st.markdown("---")
            st.markdown("#### 📊 Soil posterior — Bayesian Model Averaging")
            bma_c1, bma_c2 = st.columns([1, 2])
            with bma_c1:
                st.metric("Recommended soil prob.",
                          f"{bma_res.dominant_prob * 100:.1f}%")
                st.metric("Effective # of models",
                          f"{bma_res.n_effective_models:.1f} / 12")
                st.write(f"Confidence: **{bma_res.confidence_label}**")
                if bma_res.n_effective_models >= 4:
                    st.info("💡 High effective-model count → soil identification "
                            "is genuinely uncertain in this dataset.")
            with bma_c2:
                from bma import bma_summary_table
                bma_df = bma_summary_table(bma_res)
                bma_top = bma_df.head(6)
                fig_bma = go.Figure(go.Bar(
                    x=bma_top["사후확률(%)"].values,
                    y=bma_top["토양"].values,
                    orientation="h",
                    marker_color=[
                        "#10B981" if v >= 30 else ("#F59E0B" if v >= 15 else "#9CA3AF")
                        for v in bma_top["사후확률(%)"].values
                    ],
                    text=[f"{v:.1f}%" for v in bma_top["사후확률(%)"].values],
                    textposition="outside",
                ))
                fig_bma.update_layout(
                    title="Posterior probability P(M_k | D)",
                    xaxis_title="Posterior probability (%)",
                    yaxis=dict(autorange="reversed"),
                    height=280, margin=dict(l=10, r=10, t=40, b=30),
                )
                st.plotly_chart(fig_bma, use_container_width=True, theme=None)
            st.caption(
                "BMA reports how well each USDA soil class explains the observed "
                "data as a probability (Hoeting et al. 1999); the recharge metric "
                "uses the recommended soil's simulation."
            )


# ---------------------------------------------------------------------------
# Main chart
# ---------------------------------------------------------------------------
def _render_main_chart(result, ctx, rr_v27):
    days = np.arange(len(result["ho"]))
    ho = np.array(result["ho"], dtype=float)
    hs_kf = np.array(result["hs_kf"], dtype=float)
    po = np.array(result.get("po_shifted", result["po"]), dtype=float)
    pump_mask = np.array(result.get("pump_mask", [0]*len(ho))).astype(bool)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if pump_mask.any():
        shade_pump_plotly(fig, days, pump_mask)
    rain_mm = po * 1000
    fig.add_trace(go.Bar(
        x=days, y=rain_mm, name="Rain (mm)",
        marker=dict(color=C["rain"], opacity=0.35, line=dict(width=0))),
        secondary_y=True)
    fig.add_trace(go.Scatter(
        x=days, y=ho, mode="markers", name="Observed",
        marker=dict(color=C["observed"], size=4, opacity=0.55)),
        secondary_y=False)
    fig.add_trace(go.Scatter(
        x=days, y=hs_kf, mode="lines", name="Kalman",
        line=dict(color=C["kalman"], width=2.5)), secondary_y=False)
    if ctx.show_pure and "hs_pure" in result:
        hs_pure = np.array(result["hs_pure"], dtype=float)
        fig.add_trace(go.Scatter(
            x=days, y=hs_pure, mode="lines", name="Pure WTF",
            line=dict(color="#F59E0B", width=2.0, dash="dash")),
            secondary_y=False)

    title = (
        f"Water level — k={result.get('opt_k', ctx.k_val):.4f}, "
        f"z={result.get('opt_z', ctx.z_val):.2f}, "
        f"recharge={rr_v27:.2f}%"
    )
    fig.update_layout(title=title, height=420, hovermode="x unified",
                      margin=dict(l=40, r=20, t=50, b=40))
    fig.update_xaxes(title="Day index", gridcolor=C["grid"])
    fig.update_yaxes(title="GW level (m)", secondary_y=False, gridcolor=C["grid"])
    rain_max = float(np.nanmax(rain_mm)) if len(rain_mm) > 0 else 10
    fig.update_yaxes(title="Rain (mm)", range=[rain_max * 3.5, 0], secondary_y=True)
    st.plotly_chart(fig, use_container_width=True, theme=None)


# ---------------------------------------------------------------------------
# Bayesian Sy posterior (Phase 1)
# ---------------------------------------------------------------------------
def _render_bayesian_sy(result, ctx):
    st.markdown("---")
    st.markdown("#### 🎲 Bayesian Sy / recharge posterior (Phase 1)")
    st.caption(
        "Combines a soil-texture prior (HSG + aquifer type, Carsel-Parrish "
        "1988) with the WTF-derived effective Sy as likelihood. Pumping-test "
        "Sy values, when supplied, enter as a strong likelihood."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        hsg = st.selectbox("Surface HSG", ["A", "B", "C", "D"], index=0,
                           key="bayes_hsg_en")
    with col2:
        aq = st.selectbox("Aquifer type", ["alluvial", "bedrock"], index=0,
                          key="bayes_aq_en",
                          help="alluvial = unconsolidated; bedrock = fractured/weathered")
    with col3:
        pump_input = st.text_input("Pump-test Sy (optional)", value="",
                                   key="bayes_pump_en",
                                   help="e.g. 0.18 — leave blank if unavailable")
        try:
            pump_sy = float(pump_input) if pump_input.strip() else None
            if pump_sy is not None and not (0.001 < pump_sy < 0.5):
                st.error("Sy must be in (0.001, 0.5)")
                pump_sy = None
        except ValueError:
            pump_sy = None
            if pump_input.strip():
                st.error("Numeric input required")

    if not st.button("🎲 Run Bayesian inference", key="run_bayes_en"):
        st.info("Click the button to compute the Sy and recharge posterior.")
        return

    try:
        from bayes_sy import from_result_v27 as bayes_from_v27
        with st.spinner("Importance sampling…"):
            br = bayes_from_v27(result, hsg=hsg, aquifer=aq,
                                pump_test_sy=pump_sy, n_samples=10000)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Sy posterior**")
            st.metric("Sy (posterior mean)", f"{br.sy_post_mean:.3f}",
                      delta=f"{br.sy_post_mean - br.sy_prior_mean:+.3f} vs prior")
            st.caption(
                f"95% CI [{br.sy_post_lo95:.3f}, {br.sy_post_hi95:.3f}] · "
                f"prior μ={br.sy_prior_mean:.3f} ± σ={br.sy_prior_sd:.3f} · "
                f"sn={br.sn_used}"
            )
        with col_b:
            st.markdown("**Recharge posterior**")
            if np.isfinite(br.rech_pct_post_mean):
                st.metric("Recharge (posterior mean)",
                          f"{br.rech_pct_post_mean:.2f} %")
                st.caption(
                    f"95% CI [{br.rech_pct_post_lo95:.2f}, "
                    f"{br.rech_pct_post_hi95:.2f}] % · σ={br.rech_pct_post_sd:.2f}"
                )

        st.caption(
            f"📊 Effective sample size: **{br.n_eff:.0f}** / {br.n_samples} "
            f"({'✅ converged' if br.converged else '⚠️ ESS<100, low confidence'})"
        )

        try:
            import matplotlib.pyplot as plt
            from scipy.stats import truncnorm
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
            x = np.linspace(0.01, 0.45, 200)
            a = (0.01 - br.sy_prior_mean) / br.sy_prior_sd
            b = (0.45 - br.sy_prior_mean) / br.sy_prior_sd
            prior_pdf = truncnorm.pdf(x, a, b, loc=br.sy_prior_mean,
                                       scale=br.sy_prior_sd)
            ax = axes[0]
            ax.plot(x, prior_pdf, "--", color="#6c757d", label="Prior")
            ax.axvline(br.sy_post_mean, color="#dc2626", linewidth=2,
                       label=f"Posterior μ={br.sy_post_mean:.3f}")
            ax.axvspan(br.sy_post_lo95, br.sy_post_hi95, color="#dc2626",
                       alpha=0.15, label="95% CI")
            ax.set_xlabel("Sy"); ax.set_ylabel("density")
            ax.set_title("Sy posterior vs prior")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)

            ax2 = axes[1]
            if np.isfinite(br.rech_pct_post_mean):
                ax2.axvline(br.rech_pct_post_mean, color="#0891b2",
                            linewidth=2, label=f"Posterior μ={br.rech_pct_post_mean:.1f}%")
                ax2.axvspan(br.rech_pct_post_lo95, br.rech_pct_post_hi95,
                            color="#0891b2", alpha=0.15, label="95% CI")
                ax2.set_xlabel("Recharge (% of P)")
                ax2.set_title("Recharge posterior")
                ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.caption(f"Histogram render failed: {e}")

        st.session_state["bayes_sy_result"] = br
    except Exception as e:
        st.error(f"Bayesian inference failed: {e}")
        import traceback
        st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Save for watershed (cached mode in Tab 10)
# ---------------------------------------------------------------------------
def _render_save_for_watershed(result, ctx, has_pump_corr):
    import os
    st.markdown("---")
    st.markdown("#### 💾 Save for watershed analysis")
    st.caption(
        "Saving here lets **Tab 10 (Watershed recharge)** reuse this "
        "tuned single-well result without re-running it."
    )
    try:
        from wells_registry import WELLS
        registered = list(WELLS.keys())
    except Exception:
        registered = []

    upl = st.session_state.get("uploaded_name", "") or ""
    base = os.path.splitext(upl)[0] if upl else ""
    default = base if base in registered else (
        registered[0] if registered else base or "well1"
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        if registered:
            options = registered + ["(register new)"]
            try:
                idx = options.index(default)
            except ValueError:
                idx = 0
            choice = st.selectbox("Well name (registered)", options=options,
                                   index=idx, key="save_well_en")
        else:
            choice = "(register new)"
            st.info("No wells registered yet. New well will be created.")

        is_new = (choice == "(register new)")
        if is_new:
            new_name = st.text_input("New well name", value=default,
                                      key="new_well_en")
            sub1, sub2, sub3 = st.columns(3)
            with sub1:
                new_lat = st.number_input("Latitude (WGS84)",
                                            min_value=33.0, max_value=39.0,
                                            value=36.35, step=0.0001,
                                            format="%.4f", key="new_lat_en")
                new_lon = st.number_input("Longitude (WGS84)",
                                            min_value=124.0, max_value=132.0,
                                            value=127.37, step=0.0001,
                                            format="%.4f", key="new_lon_en")
            with sub2:
                new_ws = st.text_input("Watershed name", placeholder="e.g. Gamcheon",
                                       key="new_ws_en")
                new_aq = st.selectbox("Aquifer", ["alluvial", "bedrock"],
                                       index=1, key="new_aq_en")
            with sub3:
                new_kma = st.number_input("KMA ASOS station ID",
                                           min_value=100, max_value=300,
                                           value=133, step=1, key="new_kma_en",
                                           help="Daejeon=133, Chupungnyeong=135")
            well_name = new_name
        else:
            well_name = choice

    with col2:
        if st.button("💾 Save", type="primary", use_container_width=True,
                     key="btn_save_en"):
            try:
                from well_results_store import from_result_v27, save
                import wells_registry as wr
                if is_new:
                    if not well_name.strip():
                        st.error("Well name required")
                        return
                    if not new_ws.strip():
                        st.error("Watershed name required")
                        return
                    wr.add_well(well_name.strip(), new_lat, new_lon,
                                 new_ws.strip(), aquifer=new_aq,
                                 nearest_kma=int(new_kma), overwrite=True)
                aquifer = hydro_type = soil_code = None
                lat = lon = None
                if well_name in wr.WELLS:
                    info = wr.WELLS[well_name]
                    aquifer = info.aquifer; lat, lon = info.lat, info.lon
                    try:
                        from shp_soil_mapper import query_point
                        sq = query_point(well_name, info.lat, info.lon)
                        hydro_type = sq.hydro_type
                        soil_code = sq.soil_code
                    except Exception:
                        pass

                stored = from_result_v27(
                    well_name=well_name, result_v27=result,
                    file_path=st.session_state.get("uploaded_tmp_path", ""),
                    sn_idx=int(ctx.sn_idx),
                    soil_name=SOIL_NAMES[int(ctx.sn_idx) - 1] if 1 <= int(ctx.sn_idx) <= 12 else None,
                    pump_corrected=has_pump_corr,
                    aquifer=aquifer, hydro_type=hydro_type,
                    soil_code=soil_code, lat=lat, lon=lon,
                )
                br = st.session_state.get("bayes_sy_result")
                if br is not None:
                    stored.bayes_sy_post_mean = br.sy_post_mean
                    stored.bayes_sy_post_sd = br.sy_post_sd
                    stored.bayes_sy_post_lo95 = br.sy_post_lo95
                    stored.bayes_sy_post_hi95 = br.sy_post_hi95
                    if np.isfinite(br.rech_pct_post_mean):
                        stored.bayes_rech_pct_post_mean = br.rech_pct_post_mean
                        stored.bayes_rech_pct_post_lo95 = br.rech_pct_post_lo95
                        stored.bayes_rech_pct_post_hi95 = br.rech_pct_post_hi95
                    stored.bayes_n_eff = br.n_eff
                    stored.pump_test_sy = br.pump_test_sy
                path = save(stored)
                tag = "Newly registered + saved" if is_new else "Saved"
                st.success(f"✅ {tag}: `{path}`")
                st.caption(
                    f"Recharge {stored.recharge_ratio_pct:.2f}% · "
                    f"sn={stored.sn_idx} · P={stored.P_annual_mm:.0f} mm/yr"
                )
            except Exception as e:
                st.error(f"Save failed: {e}")

    try:
        from well_results_store import list_stored
        saved = list_stored()
        if saved:
            with st.expander(f"📂 Currently saved wells ({len(saved)})",
                              expanded=False):
                for s in saved:
                    st.write(
                        f"- **{s.well_name}** — {s.recharge_ratio_pct:.2f}% "
                        f"(sn={s.sn_idx}, {s.analyzed_at})"
                    )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# AI hydrogeological opinion (English prompt)
# ---------------------------------------------------------------------------
def _render_ai_opinion_en(result, ctx, rr_v27, rr_corr, has_pump_corr, sy_eff):
    st.markdown("---")
    if st.button("🧠 Request AI hydrogeological commentary", key="ai_opinion_en"):
        if not ctx.api_key:
            st.warning("⚠️ Enter an OpenAI API key in the sidebar first.")
            return
        with st.spinner("Drafting commentary…"):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=ctx.api_key)
                soil_name = SOIL_NAMES[int(ctx.sn_idx) - 1]
                pr_now = st.session_state.get("pump_result")
                pump_txt = ""
                if pr_now:
                    pump_txt = (
                        f"\n[Pumping pre-processing]\n"
                        f"- Pumping contamination ratio: {pr_now['pump_fraction']*100:.1f}%\n"
                        f"- Detected events: {pr_now['n_events']}\n"
                        f"- Post-correction RMSE: {pr_now['corrected']['rmse']:.4f} m\n"
                        f"- Post-correction recharge: {pr_now['corrected']['rech_rate']:.2f}%\n"
                        f"- Pre-correction recharge: {pr_now['raw']['rech_rate']:.2f}%"
                    )
                prompt = f"""You are a hydrogeologist reviewing a Water Table
Fluctuation (WTF) analysis. Provide a concise (3-paragraph) interpretation
in English that addresses (i) data quality and Kalman fit, (ii) the
recharge estimate in context of typical Korean basin recharge ranges
(12–25% of P), and (iii) any caveats about pumping contamination or
specific yield uncertainty.

[Analysis result]
- Recharge (v27 WTF): {rr_v27:.2f}%
{f"- Recharge (corrected): {rr_corr:.2f}%" if has_pump_corr else ""}
- RMSE: {float(result['rmse']):.4f} m
- Correlation coefficient (CC): {float(result['cc']):.4f}
- NSE: {result.get('nse', 'N/A')}
- KGE: {result.get('kge', 'N/A')}
- PBIAS: {result.get('pbias', 'N/A')}%
- Recommended soil: {soil_name}
- Effective Sy: {sy_eff:.4f}
- Pumping contamination index: {float(result.get('pump_contam_idx', 0)):.2f}
{pump_txt}

Be specific and quantitative. Reference Healy (2010) where appropriate."""
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=900,
                )
                ai_text = response.choices[0].message.content
                st.markdown("**AI commentary**")
                st.markdown(ai_text)
            except Exception as e:
                st.error(f"AI request failed: {e}")
