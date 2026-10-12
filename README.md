# NeuroDefect

**A Hybrid Neuro-Fuzzy-GA Decision Support System for Software Defect Prediction and Bug-Severity Assessment**

NeuroDefect is an explainable research MVP. It loads a NASA PROMISE-style software-metric dataset, performs leakage-safe preprocessing, selects features with a genetic algorithm, trains a feed-forward neural defect predictor, and derives a **risk severity assessment** with a small, documented fuzzy rule base.

NASA PC5 / JM1 / CM1 provide **binary defect labels**. They do not provide reliable four-class ground-truth severity. This system therefore reports:

- Defect Prediction (defective / non-defective)
- Defect probability from the trained neural model
- Risk Severity Assessment (derived, not ground-truth bug severity)
- Testing priority and the fuzzy rules that fired

## Architecture

```
Dataset → Validate → Clean → Stratified split
      → Fit preprocessing on training data only
      → SMOTE on training data only
      → GA feature selection on training data only
      → BPN neural predictor
      → Fuzzy risk engine
      → Streamlit decision dashboard
```

Data leakage is treated as a defect. Scalers, SMOTE, GA, and early-stopping validation never see the test set.

## Dataset requirements

A CSV (or XLSX) with numeric software metrics and a binary defect column. Common NASA names such as `defects`, `bug`, or `label` are detected automatically. If detection is ambiguous, the UI asks you to choose the column.

A bundled **synthetic CM1-like sample** lives at `data/sample/NASA_CM1_sample.csv` so the app runs without a download. For research experiments, place a real NASA file (PC5, JM1, CM1, …) in `data/raw/`.

The sample is generated from published CM1-like marginal statistics. It is **not** the official NASA distribution.

## Installation

Python 3.11 or 3.12 is specified for the research stack. The project also runs on newer CPython if wheels are available. PyTorch is preferred for the BPN; a NumPy implementation of the same architecture is used when `torch` is not installed.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python scripts/generate_cm1_sample.py
```

Optional:

```bash
pip install torch               # BPN backend
```

## Running the application

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically http://localhost:8501).

### Training workflow

1. Dataset Analysis — inspect dimensions, missingness, imbalance, and correlations.
2. Model Training — choose split, GA parameters, SMOTE, then **Train Model Engine**.
3. Progress shows generation / epoch counts. The UI does not freeze without feedback.

### Prediction workflow

1. Open Predictions.
2. Select a held-out module or override LOC / complexity.
3. The page shows defect probability, derived risk tier, feature contributions, fired rules, and a testing recommendation generated from those results.

## GA methodology

Each chromosome is a binary mask over training features. Fitness is cross-validated F1 (or MCC) of a logistic probe, minus a complexity penalty:

`fitness = model_score − λ × (selected / total)`

Selection runs **only on SMOTE-resampled training data**. The test set is untouched.

## Neuro-fuzzy methodology

1. The BPN maps GA-selected, scaled metrics to a defect probability.
2. Interpretable software metrics (complexity, LOC, branch count, Halstead volume / difficulty / effort — whichever exist) are fuzzified with **training-set quantile** triangles (Low / Medium / High).
3. Defect probability uses documented sets: Unlikely, Possible, Likely.
4. A Mamdani rule base (~14 rules, each with a rationale) produces a continuous risk score in `[0, 100]`, mapped to LOW / MEDIUM / HIGH / CRITICAL.

This is not “a neural net sitting next to a fuzzy system.” Probability and metric memberships jointly drive inference, and membership parameters are isolated so they can later be tuned by learning.

## Evaluation metrics

Reported on the **held-out test set**: Accuracy (secondary), Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, G-mean, confusion matrix (false negatives highlighted), and a 10% testing-budget recall comparison.

Do not treat training curves as test performance.

## Project layout

See `src/` for data, models, GA, fuzzy, evaluation, and the training pipeline. `app.py` only handles Streamlit navigation, session state, and calls into those modules. Styles live in `assets/styles.css`.

## Tests

```bash
pytest -q
```

The suite covers loading, duplicate/missing handling, leakage-safe scaling and SMOTE, GA chromosome validity, fuzzy membership/risk range, and prediction outputs.

## Known limitations (MVP)

- Risk severity is **derived**, not labelled by NASA.
- GA uses a logistic probe for fitness, not a full BPN inside every generation (computational safety).
- Default GA population/generations are MVP-sized. Use `mode: RESEARCH` in `config.yaml` or Settings for larger runs.
- PSO, SOM, SHAP, and Eclipse JDT/PDE severity labels are roadmap items, not MVP claims.
- The bundled sample is synthetic CM1-like data.

## Research roadmap

Phase 2: PSO selector, GA vs PSO, multi-seed ablation.  
Phase 3: PC5 ↔ JM1 cross-project validation.  
Phase 4: Eclipse JDT/PDE ground-truth severity.  
Phase 5: Friedman / Wilcoxon significance tests.  
Phase 6: SHAP and feature stability.  
Phase 7: FastAPI + React production split.

## Citation / references

- NASA PROMISE software defect datasets (CM1, JM1, PC5, …)
- Halstead, McCabe software science / complexity metrics
- Chawla et al., SMOTE
- DEAP genetic algorithms
- Mamdani fuzzy inference

Faculty question this MVP is built to answer: *Where did this prediction come from?* The UI can show the selected features, the neural probability, the high risk metrics, the fuzzy rules that fired, why the severity band was assigned, and which configuration produced the run.
