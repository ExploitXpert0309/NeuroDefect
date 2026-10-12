from __future__ import annotations

import streamlit as st

from src.evaluation.comparison import metrics_table
from src.evaluation.testing_budget import testing_budget_table
from src.ui.charts import grouped_metrics, roc_chart
from src.ui.components import empty_note, html, html_table, page_header, plotly, section


def render() -> None:
    page_header("Neurodefect / Model Comparison", "Performance Benchmark Matrix")
    result = st.session_state.get("result")
    if result is None:
        empty_note(
            "Train the engine to compare Majority, Logistic Regression, BPN, and Neuro-Fuzzy-GA on the held-out test set."
        )
        return

    table = metrics_table(result.metrics)
    with section("Performance Metric Benchmarks", key="cmp_metrics"):
        rows = []
        for i, row in table.iterrows():
            mcc = row["Matthews CC"]
            mcc_html = (
                f"<span style='color:#16a34a;font-weight:600'>{mcc:.3f}</span>"
                if i == 0 or row["Classifier Model Architecture"] == "Neuro-Fuzzy-GA"
                else f"{mcc:.3f}"
            )
            rows.append(
                [
                    f"<b>{row['Classifier Model Architecture']}</b>"
                    if row["Classifier Model Architecture"] == "Neuro-Fuzzy-GA"
                    else row["Classifier Model Architecture"],
                    f"{row['Accuracy']:.3f}",
                    f"{row['Precision']:.3f}",
                    f"{row['Recall']:.3f}",
                    f"{row['F1-Score']:.3f}",
                    mcc_html,
                ]
            )
        hybrid = [r for r in rows if "Neuro-Fuzzy-GA" in r[0]]
        rest = [r for r in rows if "Neuro-Fuzzy-GA" not in r[0]]
        html(
            html_table(
                ["Classifier Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "Matthews CC"],
                hybrid + rest,
                highlight_first=True,
            )
        )
        st.caption(
            "Accuracy is shown as a secondary metric. F1, PR-AUC, MCC, and G-mean are the primary research metrics because defect datasets are imbalanced."
        )

    left, right = st.columns(2, gap="small")
    with left:
        with section("Primary Metrics Grouped Comparison", key="cmp_grouped"):
            plotly(grouped_metrics(table))
    with right:
        with section("ROC Response Curves", key="cmp_roc"):
            aucs = {name: m.roc_auc for name, m in result.metrics.items()}
            plotly(roc_chart(result.roc, aucs))

    hybrid_metrics = result.metrics.get("Neuro-Fuzzy-GA")
    if hybrid_metrics:
        with section("Confusion Matrix (Neuro-Fuzzy-GA, test set)", key="cmp_cm"):
            html(
                html_table(
                    ["", "Predicted clean", "Predicted defective"],
                    [
                        ["Actual clean", str(hybrid_metrics.tn), str(hybrid_metrics.fp)],
                        [
                            "Actual defective",
                            f"<span style='color:#dc2626;font-weight:700'>{hybrid_metrics.fn}</span>",
                            str(hybrid_metrics.tp),
                        ],
                    ],
                )
            )
            st.caption(
                "False negatives are highlighted because missing a defective module is typically more costly than extra review."
            )

    with section("Testing Budget (10% of modules reviewed)", key="cmp_budget"):
        budget = testing_budget_table(result.y_test, result.scores, budget_frac=0.10, random_state=result.seed)
        budget_rows = [
            [r["Strategy"], r["Budget"], f"{r['Defects found (share of actual defective modules)'] * 100:.1f}%"]
            for _, r in budget.iterrows()
        ]
        html(html_table(["Strategy", "Budget", "Share of actual defective modules found"], budget_rows))
        st.caption("Computed from held-out test scores. Random selection is a baseline, not a trained model.")
