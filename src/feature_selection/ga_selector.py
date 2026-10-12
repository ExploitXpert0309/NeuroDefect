from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
from deap import algorithms, base, creator, tools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from src.feature_selection.base import FeatureSelector, ProgressCallback, SelectionResult
from src.utils.exceptions import TrainingError

_CREATOR_READY = False


def _ensure_creator() -> None:
    global _CREATOR_READY
    if _CREATOR_READY:
        return
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
    _CREATOR_READY = True


class GAFeatureSelector(FeatureSelector):
    def __init__(
        self,
        population_size: int = 16,
        generations: int = 12,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.05,
        random_state: int = 42,
        min_features: int = 4,
        max_features: int | None = None,
        fitness_metric: str = "f1",
        lambda_penalty: float = 0.15,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.random_state = random_state
        self.min_features = min_features
        self.max_features = max_features
        self.fitness_metric = fitness_metric
        self.lambda_penalty = lambda_penalty

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        progress: ProgressCallback | None = None,
    ) -> SelectionResult:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        n_features = X.shape[1]
        if n_features == 0:
            raise TrainingError("No features available for genetic algorithm selection.")
        min_feat = max(1, min(self.min_features, n_features))
        max_feat = self.max_features or n_features
        max_feat = max(min_feat, min(max_feat, n_features))

        rng = random.Random(self.random_state)
        np_rng = np.random.default_rng(self.random_state)
        _ensure_creator()

        def repair(individual: list[int]) -> list[int]:
            selected = [i for i, bit in enumerate(individual) if bit]
            if len(selected) < min_feat:
                missing = [i for i in range(n_features) if individual[i] == 0]
                np_rng.shuffle(missing)
                for idx in missing[: min_feat - len(selected)]:
                    individual[idx] = 1
            selected = [i for i, bit in enumerate(individual) if bit]
            if len(selected) > max_feat:
                np_rng.shuffle(selected)
                for idx in selected[max_feat:]:
                    individual[idx] = 0
            if sum(individual) == 0:
                individual[int(np_rng.integers(0, n_features))] = 1
            return individual

        def evaluate(individual: list[int]) -> tuple[float]:
            repair(individual)
            mask = np.array(individual, dtype=bool)
            Xs = X[:, mask]
            score = _cv_score(Xs, y, self.fitness_metric, self.random_state)
            ratio = float(mask.sum()) / n_features
            return (score - self.lambda_penalty * ratio,)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", rng.randint, 0, 1)
        toolbox.register(
            "individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=max(self.mutation_probability, 1.0 / n_features))
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=self.population_size)
        for ind in population:
            repair(ind)

        start = time.perf_counter()
        history: list[float] = []
        hof = tools.HallOfFame(1)

        invalid = [ind for ind in population if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit
        hof.update(population)
        history.append(float(hof[0].fitness.values[0]))
        if progress:
            progress(0, self.generations, _progress_payload(hof[0], n_features, history[-1]))

        for gen in range(1, self.generations + 1):
            offspring = algorithms.varAnd(
                population, toolbox, cxpb=self.crossover_probability, mutpb=1.0
            )
            for ind in offspring:
                repair(ind)
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))
            hof.update(population)
            history.append(float(hof[0].fitness.values[0]))
            if progress:
                progress(gen, self.generations, _progress_payload(hof[0], n_features, history[-1]))

        best = repair(list(hof[0]))
        mask = np.array(best, dtype=bool)
        selected = [name for name, bit in zip(feature_names, mask) if bit]
        runtime = time.perf_counter() - start
        return SelectionResult(
            selected_features=selected,
            mask=mask,
            fitness=float(hof[0].fitness.values[0]),
            history=history,
            runtime_seconds=runtime,
            metadata={
                "n_selected": int(mask.sum()),
                "n_total": n_features,
                "metric": self.fitness_metric,
                "generations": self.generations,
                "population_size": self.population_size,
            },
        )


def _progress_payload(individual: Any, n_features: int, fitness: float) -> dict[str, Any]:
    n_sel = int(sum(individual))
    return {
        "best_fitness": fitness,
        "n_selected": n_sel,
        "n_total": n_features,
    }


def _cv_score(X: np.ndarray, y: np.ndarray, metric: str, seed: int) -> float:
    splits = min(3, max(2, int(np.min(np.bincount(y)))))
    if splits < 2:
        return 0.0
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_idx, val_idx in cv.split(X, y):
        model = LogisticRegression(max_iter=250, class_weight="balanced", solver="liblinear")
        try:
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
        except Exception:
            scores.append(0.0)
            continue
        if metric == "mcc":
            scores.append(float(matthews_corrcoef(y[val_idx], pred)))
        else:
            scores.append(float(f1_score(y[val_idx], pred, zero_division=0)))
    return float(np.mean(scores)) if scores else 0.0
