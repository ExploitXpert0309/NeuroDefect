from __future__ import annotations

import streamlit as st

from src.pipeline.predictor import predict_from_raw_row
from src.ui.charts import contribution_bars, probability_donut
from src.ui.components import empty_note, html, page_header, plotly, section
from src.utils.exceptions import NeuroDefectError


def render() -> None:
    page_header("Neurodefect / Predictions", "Active Module Risk Evaluator")
    result = st.session_state.get("result")
    if result is None:
        empty_note("Please train a model before running predictions.")
        return

    with section("Inspect Module Target", key="pr_inspect"):
        ids = result.display_ids
        choice = st.selectbox("Select a held-out module", ids)
        idx = ids.index(choice)
        raw = result.X_test_raw.iloc[idx]
        loc_col = next((c for c in raw.index if c.lower() in {"loc", "locode"}), raw.index[0])
        cx_col = next((c for c in raw.index if c in {"v(g)", "branchCount"}), raw.index[min(1, len(raw.index) - 1)])
        c1, c2, c3 = st.columns((2.2, 1, 1))
        c1.caption("or manually override features:")
        loc_val = c2.number_input("LOC", value=float(raw[loc_col]), key="ovr_loc")
        cx_val = c3.number_input("Complexity", value=float(raw[cx_col]), key="ovr_cx")

    row = raw.to_dict()
    row[loc_col] = loc_val
    row[cx_col] = cx_val
    try:
        pred = predict_from_raw_row(result, row, module_id=choice)
    except NeuroDefectError as exc:
        st.error(str(exc))
        return

    with section("Fuzzy-Logic Classification Output", key="pr_output"):
        a, b = st.columns((0.7, 1.6), gap="small", vertical_alignment="center")
        with a:
            plotly(probability_donut(pred.probability))
        with b:
            color = "#ef4444" if pred.label == "DEFECTIVE" else "#16a34a"
            bg = "#fef2f2" if pred.label == "DEFECTIVE" else "#ecfdf3"
            html(
                f"<span class='nd-badge-lg' style='background:{bg};color:{color}'>{pred.label}</span>"
                f" &nbsp; Confidence Level: <b>{pred.confidence*100:.1f}%</b>"
                "<p class='nd-note'>Prediction confidence is the distance from the 0.5 decision boundary, "
                "not a statistical confidence interval.</p>"
            )
            st.caption("Evaluated Risk Tier")
            active = pred.severity.lower()
            html(
                f"""
                <div class="nd-tier">
                  <div class="{'on-low' if active == 'low' else ''}">LOW</div>
                  <div class="{'on-medium' if active == 'medium' else ''}">MEDIUM</div>
                  <div class="{'on-high' if active == 'high' else ''}">HIGH</div>
                  <div class="{'on-critical' if active == 'critical' else ''}">CRITICAL</div>
                </div>
                """
            )

    left, right = st.columns(2, gap="small")
    with left:
        with section("Feature Risk Contributions", key="pr_contrib"):
            top = pred.feature_contribs[:3]
            if top:
                plotly(contribution_bars(top))
            else:
                st.caption("No feature contributions available for this module.")
    with right:
        with section("Active Fuzzy Reasoning Nodes", key="pr_rules"):
            rules = pred.risk.fired_rules[:3]
            if not rules:
                st.caption("No rules fired above the minimum strength threshold.")
            for rule in rules:
                html(
                    f"<div class='nd-rule'><div>{rule.description[:72]}{'…' if len(rule.description) > 72 else ''}</div>"
                    f"<span>Strength: {rule.strength:.2f}</span></div>"
                )

    with section("Operational Action Protocol", key="pr_action"):
        html(
            f"<div class='nd-alert critical'><strong>"
            f"{'IMMEDIATE CODE REVIEW REQUIRED' if pred.severity in {'HIGH', 'CRITICAL'} else 'TESTING RECOMMENDATION'}"
            f"</strong><br>{pred.risk.recommendation}</div>"
        )
        st.caption(
            "This recommendation is generated from the trained neural probability and the fuzzy rule base. "
            "Derived risk severity is not a ground-truth NASA bug-severity label."
        )
