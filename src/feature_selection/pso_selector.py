from __future__ import annotations

from src.feature_selection.base import FeatureSelector, ProgressCallback, SelectionResult


class PSOFeatureSelector(FeatureSelector):
    """Placeholder for the Phase 2 research roadmap. Not used in the MVP."""

    def fit(
        self,
        X,
        y,
        feature_names: list[str],
        progress: ProgressCallback | None = None,
    ) -> SelectionResult:
        raise NotImplementedError(
            "PSO feature selection is on the research roadmap (Phase 2). "
            "Use GAFeatureSelector for the MVP."
        )
