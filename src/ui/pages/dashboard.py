from __future__ import annotations

from collections import Counter

import streamlit as st

from src.pipeline.predictor import recent_predictions
from src.ui.charts import convergence_chart, donut, risk_bars
from src.ui.components import empty_note, html, html_table, kpi_card, page_header, plotly, risk_chip, section


def render() -> None:
    page_header("Neurodefect / Home", "Decision Support Dashboard")
    dataset = st.session_state.get("dataset")
    result = st.session_state.get("result")

    if dataset is None:
        empty_note("Load a NASA defect dataset from Dataset Analysis or Model Training to begin.")
        return

    quality = result.quality if result else st.session_state.get("quality")
    n_rows = quality.n_rows if quality else len(dataset.frame)
    defect_rate = quality.defect_rate if quality else 0.0

    selected = f"{len(result.selected_features)}/{len(result.all_features)}" if result else "—"
    status = "Trained" if result else "Idle"
    status_chip = "green" if result else "slate"
    status_meta = "Active" if result else "Awaiting training"

    k1, k2, k3, k4 = st.columns(4, gap="small")
    k1.markdown(kpi_card("Total Modules", f"{n_rows}", dataset.name.replace("_", " "), "slate"), unsafe_allow_html=True)
    k2.markdown(
        kpi_card("Defect Rate", f"{defect_rate * 100:.1f}%", f"{defect_rate * 100:.1f}% base" if quality else "", "amber"),
        unsafe_allow_html=True,
    )
    k3.markdown(
        kpi_card("Selected Features", selected, "GA Optimized" if result else "Not selected", "blue"),
        unsafe_allow_html=True,
    )
    k4.markdown(kpi_card("Model Status", status, status_meta, status_chip), unsafe_allow_html=True)

    left, right = st.columns((1.7, 1.0), gap="small")
    with left:
        with section("Genetic-Neuro-Fuzzy Optimizer Convergence", key="dash_convergence", height="stretch"):
            if result:
                acc = result.bpn_history.train_acc or result.ga_history
                f1 = result.bpn_history.train_f1 or None
                fig = convergence_chart(acc, f1)
            else:
                fig = convergence_chart()
            plotly(fig, height="stretch")

    with right:
        with st.container(height="stretch", gap="small"):
            with section("Defect Distribution", key="dash_defect"):
                if quality:
                    d1, d2 = st.columns((1.0, 1.05), gap="small", vertical_alignment="center")
                    with d1:
                        plotly(donut(quality.n_clean, quality.n_defective))
                    with d2:
                        clean_pct = (1 - quality.defect_rate) * 100
                        def_pct = quality.defect_rate * 100
                        html(
                            f"""
                            <div class="nd-dist-stats">
                              <div class="nd-dist-row">
                                <span class="nd-dist-dot clean"></span>
                                <div>
                                  <div class="nd-dist-label">Clean</div>
                                  <div class="nd-dist-value">{quality.n_clean:,}</div>
                                  <div class="nd-dist-pct">{clean_pct:.1f}%</div>
                                </div>
                              </div>
                              <div class="nd-dist-row">
                                <span class="nd-dist-dot defective"></span>
                                <div>
                                  <div class="nd-dist-label">Defective</div>
                                  <div class="nd-dist-value">{quality.n_defective:,}</div>
                                  <div class="nd-dist-pct def">{def_pct:.1f}%</div>
                                </div>
                              </div>
                            </div>
                            """
                        )
                else:
                    st.caption("Load a dataset to see class balance.")

            with section("Risk Severity Distribution", key="dash_risk", height="stretch"):
                if result:
                    plotly(risk_bars(_risk_counts(result)), height="stretch")
                else:
                    st.caption("Risk severity is derived after a model is trained.")

    with section("Recent Software Module Predictions", key="dash_predictions"):
        if result:
            recents = recent_predictions(result, limit=5)
            rows = []
            for item in recents:
                color = "#ef4444" if item.probability >= 0.6 else "#0f172a"
                rows.append(
                    [
                        item.module_id,
                        f"<span style='color:{color};font-weight:600'>{item.probability * 100:.1f}%</span>",
                        risk_chip(item.severity),
                        f"{item.confidence * 100:.1f}%",
                        item.inspection_date,
                    ]
                )
            html(
                html_table(
                    ["Module ID", "Defect Probability", "Risk Severity", "Confidence", "Inspection Date"],
                    rows,
                )
            )
            st.caption(
                "Risk severity is a derived assessment from defect probability, software metrics, and fuzzy inference — "
                "not a NASA ground-truth bug-severity label."
            )
        else:
            st.caption("Predictions appear here after the hybrid model has been trained.")


def _risk_counts(result) -> dict[str, int]:
    from src.pipeline.predictor import predict_module

    counts: Counter[str] = Counter()
    n = min(len(result.X_test_raw), 80)
    for i in range(n):
        pred = predict_module(result, i)
        counts[pred.severity] += 1
    return dict(counts)
