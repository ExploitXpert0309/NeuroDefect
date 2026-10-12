from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data.cleaner import CleaningLog, clean_frame
from src.data.imbalance import SmoteReport, apply_smote
from src.data.loader import LoadedDataset
from src.data.preprocessing import FittedPreprocessor, fit_preprocessor
from src.data.quality import DatasetQualityReport, build_quality_report
from src.data.splitting import SplitResult, stratified_split
from src.data.validator import validate_target
from src.evaluation.metrics import ClassificationMetrics, classification_metrics, roc_points
from src.explainability.explainer import ga_feature_ranks
from src.feature_selection.ga_selector import GAFeatureSelector
from src.fuzzy.risk_engine import FuzzyRiskEngine
from src.models.baselines import make_sklearn_baselines
from src.models.bpn import BPNClassifier, TrainHistory
from src.pipeline.artifacts import new_run_dir, save_run
from src.utils.config import AppConfig, ROOT
from src.utils.exceptions import DataError, TrainingError
from src.utils.reproducibility import set_seed

ProgressFn = Callable[[str, float, str, dict[str, Any]], None]


@dataclass
class TrainingResult:
    dataset_name: str
    target_column: str
    quality: DatasetQualityReport
    cleaning: CleaningLog
    split: SplitResult
    preprocessor: FittedPreprocessor
    smote: SmoteReport
    selected_features: list[str]
    all_features: list[str]
    ga_history: list[float]
    ga_fitness: float
    ga_runtime: float
    bpn_history: TrainHistory
    metrics: dict[str, ClassificationMetrics]
    roc: dict[str, tuple[np.ndarray, np.ndarray]]
    scores: dict[str, np.ndarray]
    y_test: np.ndarray
    X_test_raw: pd.DataFrame
    X_test_scaled: pd.DataFrame
    display_ids: list[str]
    models: dict[str, Any]
    bpn: BPNClassifier
    fuzzy: FuzzyRiskEngine
    feature_ranks: list[tuple[str, float]]
    duration: float
    seed: int
    mode: str
    run_dir: str
    logs: list[str] = field(default_factory=list)
    leakage_guards: dict[str, Any] = field(default_factory=dict)


def train_pipeline(
    dataset: LoadedDataset,
    config: AppConfig,
    progress: ProgressFn | None = None,
) -> TrainingResult:
    if dataset.target_column is None:
        raise DataError("Target column was not found. Please select the defect label column.")

    def report(phase: str, frac: float, message: str, extra: dict | None = None) -> None:
        if progress:
            progress(phase, frac, message, extra or {})

    seed = int(config.training().get("random_state", 42))
    set_seed(seed)
    started = time.perf_counter()
    logs: list[str] = []

    report("validate", 0.04, "Validating target column and cleaning the dataset")
    target = validate_target(dataset.frame[dataset.target_column])
    aligned = dataset.frame.copy()
    aligned[dataset.target_column] = target
    features, y, cleaning, id_series = clean_frame(aligned, dataset.target_column)
    y = validate_target(y)
    logs.extend(cleaning.lines())
    if features.shape[1] < 2:
        raise TrainingError("Fewer than two numeric features remain after cleaning.")

    quality = build_quality_report(
        dataset.frame, dataset.target_column, dataset.name, list(features.columns)
    )

    report("split", 0.10, "Creating a stratified train/test split")
    test_size = float(config.training().get("test_size", 0.3))
    split = stratified_split(
        features, y.astype(int), test_size=test_size, random_state=seed, ids=id_series
    )
    logs.append(
        f"Train/test split: {len(split.X_train)} / {len(split.X_test)} "
        f"(stratified, test_size={test_size:.0%})"
    )

    report("preprocess", 0.18, "Fitting preprocessing on training data only")
    preprocessor = fit_preprocessor(split.X_train)
    X_train = preprocessor.transform(split.X_train)
    X_test = preprocessor.transform(split.X_test)
    y_train = split.y_train.astype(int)
    y_test = split.y_test.astype(int)

    report("smote", 0.26, "Applying SMOTE to training data only")
    smote_cfg = config.smote()
    X_res, y_res, smote = apply_smote(
        X_train,
        y_train,
        enabled=bool(smote_cfg.get("enabled", True)),
        random_state=seed,
        k_neighbors=int(smote_cfg.get("k_neighbors", 5)),
    )
    logs.append(
        f"Before SMOTE — defective: {smote.before_defective}, non-defective: {smote.before_clean}"
    )
    logs.append(
        f"After SMOTE — defective: {smote.after_defective}, non-defective: {smote.after_clean}"
    )
    logs.append("Test distribution was not resampled.")

    ga_cfg = config.ga()
    report("ga", 0.34, "Running genetic algorithm feature selection on training data")

    def ga_progress(gen: int, total: int, payload: dict[str, Any]) -> None:
        frac = 0.34 + 0.28 * (gen / max(total, 1))
        report(
            "ga",
            frac,
            f"Generation {gen} / {total}  ·  Best fitness {payload['best_fitness']:.3f}  ·  "
            f"Features selected: {payload['n_selected']} / {payload['n_total']}",
            payload,
        )

    selector = GAFeatureSelector(
        population_size=int(ga_cfg.get("population_size", 16)),
        generations=int(ga_cfg.get("generations", 12)),
        crossover_probability=float(ga_cfg.get("crossover_probability", 0.8)),
        mutation_probability=float(ga_cfg.get("mutation_probability", 0.05)),
        random_state=int(ga_cfg.get("random_state", seed)),
        min_features=int(ga_cfg.get("min_features", 4)),
        max_features=ga_cfg.get("max_features"),
        fitness_metric=str(ga_cfg.get("fitness_metric", "f1")),
        lambda_penalty=float(ga_cfg.get("lambda_penalty", 0.15)),
    )
    selection = selector.fit(
        X_res.to_numpy(), y_res.to_numpy(), list(X_res.columns), progress=ga_progress
    )
    selected = selection.selected_features
    logs.append(
        f"GA selected {len(selected)} / {len(X_res.columns)} features "
        f"(fitness={selection.fitness:.3f}, {selection.runtime_seconds:.1f}s)"
    )

    X_train_sel = X_res[selected]
    X_test_sel = X_test[selected]
    X_train_all = X_res
    X_test_all = X_test

    inner_X, inner_X_val, inner_y, inner_y_val = train_test_split(
        X_train_sel,
        y_res,
        test_size=0.15,
        random_state=seed,
        stratify=y_res,
    )

    train_cfg = config.training()
    bpn_kwargs = dict(
        hidden_sizes=tuple(train_cfg.get("hidden_sizes", [64, 32])),
        dropout=float(train_cfg.get("dropout", 0.2)),
        learning_rate=float(train_cfg.get("learning_rate", 0.001)),
        optimizer=str(train_cfg.get("optimizer", "sgd_momentum")),
        momentum=float(train_cfg.get("momentum", 0.9)),
        epochs=int(train_cfg.get("epochs", 40)),
        batch_size=int(train_cfg.get("batch_size", 32)),
        patience=int(train_cfg.get("patience", 8)),
        random_state=seed,
    )

    report("bpn", 0.66, "Training the BPN neural network on GA-selected features")
    bpn_history_holder: dict[str, TrainHistory] = {}

    def bpn_cb(epoch: int, total: int, history: TrainHistory) -> None:
        bpn_history_holder["h"] = history
        acc = history.train_acc[-1] if history.train_acc else 0.0
        loss = history.train_loss[-1] if history.train_loss else 0.0
        report(
            "bpn",
            0.66 + 0.16 * (epoch / max(total, 1)),
            f"Epoch {epoch} / {total}  ·  train F1 {history.train_f1[-1]:.3f}  ·  val F1 {history.val_f1[-1]:.3f}",
            {"epoch": epoch, "total": total, "acc": acc, "loss": loss, "history": history},
        )

    bpn = BPNClassifier(**bpn_kwargs)
    bpn.fit(inner_X.to_numpy(), inner_y.to_numpy(), inner_X_val.to_numpy(), inner_y_val.to_numpy(), callback=bpn_cb)
    bpn_full = BPNClassifier(**bpn_kwargs)
    bpn_full.fit(X_train_all.to_numpy(), y_res.to_numpy())

    report("baselines", 0.84, "Training baseline classifiers on the same training split")
    fitted_models: dict[str, Any] = {}
    metrics: dict[str, ClassificationMetrics] = {}
    scores: dict[str, np.ndarray] = {}
    roc: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    baselines = make_sklearn_baselines(seed)
    for name, model in baselines.items():
        estimator = clone(model)
        if name == "Logistic Regression":
            estimator.fit(X_train_all, y_res)
            proba = estimator.predict_proba(X_test_all)[:, 1]
        else:
            estimator.fit(X_train_all, y_res)
            proba = estimator.predict_proba(X_test_all)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fitted_models[name] = estimator
        metrics[name] = classification_metrics(y_test, pred, proba)
        scores[name] = proba
        roc[name] = roc_points(y_test, proba)

    bpn_all_scores = bpn_full.predict_proba(X_test_all.to_numpy())[:, 1]
    bpn_all_pred = (bpn_all_scores >= 0.5).astype(int)
    metrics["BPN"] = classification_metrics(y_test, bpn_all_pred, bpn_all_scores)
    scores["BPN"] = bpn_all_scores
    roc["BPN"] = roc_points(y_test, bpn_all_scores)
    fitted_models["BPN"] = bpn_full

    hybrid_scores = bpn.predict_proba(X_test_sel.to_numpy())[:, 1]
    hybrid_pred = (hybrid_scores >= 0.5).astype(int)
    metrics["Neuro-Fuzzy-GA"] = classification_metrics(y_test, hybrid_pred, hybrid_scores)
    scores["Neuro-Fuzzy-GA"] = hybrid_scores
    roc["Neuro-Fuzzy-GA"] = roc_points(y_test, hybrid_scores)
    fitted_models["Neuro-Fuzzy-GA"] = bpn

    report("fuzzy", 0.92, "Fitting the fuzzy risk engine on training metrics")
    fuzzy = FuzzyRiskEngine(risk_bounds=config.fuzzy().get("risk_bounds"))
    fuzzy.fit(split.X_train)

    ranks = ga_feature_ranks(X_res, y_res, selected)
    duration = time.perf_counter() - started
    logs.append(f"Training duration: {duration:.1f}s")
    logs.append(f"Final training features: {len(selected)}")

    ids = split.ids_test or _display_ids(dataset.name, split.test_index)
    run_dir = new_run_dir(ROOT / "models")
    bundle = {
        "dataset_name": dataset.name,
        "target_column": dataset.target_column,
        "selected_features": selected,
        "all_features": list(features.columns),
        "preprocessor": preprocessor,
        "fuzzy": fuzzy.to_dict(),
        "seed": seed,
        "mode": config.mode,
        "duration": duration,
        "metrics": metrics,
        "ga_history": selection.history,
        "models": {k: v for k, v in fitted_models.items() if k != "Neuro-Fuzzy-GA"},
        "bpn": bpn,
        "cleaning": cleaning,
        "smote": smote,
    }
    save_run(run_dir, bundle)

    leakage_guards = {
        "train_index": split.train_index,
        "test_index": split.test_index,
        "smote_train_only": True,
        "ga_train_only": True,
        "scaler_train_only": True,
        "test_size": len(split.X_test),
        "smote_grew_train": smote.after_defective + smote.after_clean
        >= smote.before_defective + smote.before_clean,
    }

    report("done", 1.0, "Training complete")
    return TrainingResult(
        dataset_name=dataset.name,
        target_column=dataset.target_column,
        quality=quality,
        cleaning=cleaning,
        split=split,
        preprocessor=preprocessor,
        smote=smote,
        selected_features=selected,
        all_features=list(features.columns),
        ga_history=selection.history,
        ga_fitness=selection.fitness,
        ga_runtime=selection.runtime_seconds,
        bpn_history=bpn.history_,
        metrics=metrics,
        roc=roc,
        scores=scores,
        y_test=y_test.to_numpy(),
        X_test_raw=split.X_test.reset_index(drop=True),
        X_test_scaled=X_test_sel.reset_index(drop=True),
        display_ids=ids,
        models=fitted_models,
        bpn=bpn,
        fuzzy=fuzzy,
        feature_ranks=ranks,
        duration=duration,
        seed=seed,
        mode=config.mode,
        run_dir=str(run_dir),
        logs=logs,
        leakage_guards=leakage_guards,
    )


def _display_ids(dataset_name: str, test_index: list[int]) -> list[str]:
    prefix = "CM1" if "cm1" in dataset_name.lower() else dataset_name.split("_")[0]
    return [f"{prefix}_module_{int(i):03d}" for i in test_index]
