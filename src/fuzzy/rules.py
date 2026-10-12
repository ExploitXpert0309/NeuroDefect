from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FuzzyRule:
    rule_id: str
    antecedents: tuple[tuple[str, str], ...]  # (variable, linguistic value)
    consequent: str
    rationale: str

    def describe(self) -> str:
        parts = [f"{var} IS {val}" for var, val in self.antecedents]
        return "IF " + " AND ".join(parts) + f" THEN risk IS {self.consequent}"


def default_rule_base() -> list[FuzzyRule]:
    """Small interpretable rule base. Every rule has a documented rationale."""
    return [
        FuzzyRule(
            "R01",
            (("complexity", "HIGH"), ("effort", "HIGH")),
            "HIGH",
            "High McCabe complexity combined with high Halstead effort is associated with defect-prone modules.",
        ),
        FuzzyRule(
            "R02",
            (("complexity", "HIGH"), ("branch", "HIGH")),
            "CRITICAL",
            "Dense branching with high cyclomatic complexity increases untested path risk.",
        ),
        FuzzyRule(
            "R03",
            (("loc", "HIGH"), ("complexity", "MEDIUM")),
            "MEDIUM",
            "Large modules with moderate complexity still require extra review effort.",
        ),
        FuzzyRule(
            "R04",
            (("complexity", "LOW"), ("effort", "LOW")),
            "LOW",
            "Small, simple modules historically have lower defect density.",
        ),
        FuzzyRule(
            "R05",
            (("defect_prob", "LIKELY"), ("complexity", "HIGH")),
            "CRITICAL",
            "A high neural defect probability plus high complexity warrants immediate inspection.",
        ),
        FuzzyRule(
            "R06",
            (("defect_prob", "LIKELY"), ("volume", "HIGH")),
            "HIGH",
            "High predicted defect chance in a large Halstead volume module raises testing priority.",
        ),
        FuzzyRule(
            "R07",
            (("defect_prob", "UNLIKELY"), ("complexity", "LOW")),
            "LOW",
            "Low predicted probability and low complexity jointly support a low risk assessment.",
        ),
        FuzzyRule(
            "R08",
            (("loc", "HIGH"), ("branch", "HIGH")),
            "HIGH",
            "Large, highly branched modules are harder to cover with tests.",
        ),
        FuzzyRule(
            "R09",
            (("effort", "HIGH"), ("volume", "HIGH")),
            "HIGH",
            "High Halstead effort and volume indicate computationally dense, error-prone code.",
        ),
        FuzzyRule(
            "R10",
            (("complexity", "MEDIUM"), ("defect_prob", "POSSIBLE")),
            "MEDIUM",
            "Uncertain neural output with moderate complexity is treated as medium testing priority.",
        ),
        FuzzyRule(
            "R11",
            (("complexity", "HIGH"), ("loc", "HIGH"), ("defect_prob", "LIKELY")),
            "CRITICAL",
            "The conjunction of size, complexity, and high defect probability is the strongest risk signal.",
        ),
        FuzzyRule(
            "R12",
            (("difficulty", "HIGH"), ("complexity", "HIGH")),
            "HIGH",
            "Halstead difficulty with high complexity reflects dense operator usage and harder maintenance.",
        ),
        FuzzyRule(
            "R13",
            (("complexity", "LOW"), ("loc", "LOW"), ("defect_prob", "UNLIKELY")),
            "LOW",
            "Small, simple modules with low predicted defect chance can be deprioritized.",
        ),
        FuzzyRule(
            "R14",
            (("branch", "LOW"), ("complexity", "LOW")),
            "LOW",
            "Few branches and low complexity reduce the chance of untested defective paths.",
        ),
    ]
