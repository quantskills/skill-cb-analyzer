"""Scoring engine: 4-dimension weighted composite score (0-100) + risk penalty.

Formula:
    base = 0.40 × valuation + 0.30 × clause + 0.20 × linkage + 0.10 × structure
    with_penalty = base × 100 + risk_penalty
    final = clamp(with_penalty, 0, 100)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from core._types import safe_float

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    """Per-bond scoring breakdown."""
    bond_code: str = ""
    bond_name: str = ""
    stock_code: str = ""
    stock_name: str = ""
    cb_price: float = 0.0
    premium_rate: float = 0.0
    conversion_value: float = 0.0
    double_low: float = 0.0
    ytm: float = 0.0

    # Dimension scores (0-100)
    valuation_score: float = 0.0
    clause_score: float = 0.0
    linkage_score: float = 0.0
    structure_score: float = 0.0

    # Final
    composite_score: float = 0.0
    neutralized_score: float = 0.0
    risk_penalty: float = 0.0
    grade: str = "C"
    rank: int = 0

    # Detail
    triggered_signals: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    excluded: bool = False
    exclude_reason: str = ""

    def to_summary_dict(self) -> dict:
        """Convert to a dict suitable for CLI display and LLM analysis."""
        return {
            "rank": self.rank,
            "name": self.bond_name,
            "code": self.bond_code,
            "stock_name": self.stock_name,
            "stock_code": self.stock_code,
            "cb_price": self.cb_price,
            "premium_rate": self.premium_rate,
            "conversion_value": self.conversion_value,
            "double_low": self.double_low,
            "ytm": self.ytm * 100,  # decimal → percentage for LLM prompt display
            "valuation_score": self.valuation_score,
            "clause_score": self.clause_score,
            "linkage_score": self.linkage_score,
            "structure_score": self.structure_score,
            "score": self.composite_score,
            "neutralized_score": self.neutralized_score,
            "grade": self.grade,
            "triggered_patterns": self.triggered_signals[:3],
            "risk_flags": self.risk_flags,
        }


class Scorer:
    """4-dimension weighted scoring with risk adjustment."""

    # Default grade thresholds (overridable via config scoring.grades)
    DEFAULT_GRADES = [
        (55, "A+", "强烈关注"),
        (45, "A", "值得跟踪"),
        (40, "B+", "偏积极"),
        (37, "B", "温和"),
        (34, "C", "中性"),
        (30, "D", "偏弱"),
        (0, "E", "回避"),
    ]

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        sc = cfg.get("scoring", {})
        self._w_val = float(sc.get("valuation_weight", 0.40))
        self._w_clause = float(sc.get("clause_weight", 0.30))
        self._w_link = float(sc.get("linkage_weight", 0.20))
        self._w_struct = float(sc.get("structure_weight", 0.10))
        self._dimension_floor = float(sc.get("dimension_floor", 0.30))
        self._industry_neutralize = bool(sc.get("industry_neutralize", False))

        # Load grade thresholds from config if provided
        grades_cfg = sc.get("grades")
        if grades_cfg:
            self.grades = sorted(
                [(g["threshold"], g["grade"], g["label"]) for g in grades_cfg],
                key=lambda x: x[0],
                reverse=True,
            )
        else:
            self.grades = list(self.DEFAULT_GRADES)

    def compute(
        self,
        idx: int,
        row: pd.Series,
        val_signals: dict,
        clause_signals: dict,
        link_signals: dict,
        struct_signals: dict,
        risk_penalty: float,
        val_composite: float,
        clause_composite: float,
        link_composite: float,
        struct_composite: float,
        triggered: list[str],
        risk_flags: list[str],
    ) -> ScoreResult:
        """Compute final score for a single CB.

        Args:
            idx: Index in DataFrame.
            row: CB data row.
            val_signals: Valuation signals dict (per-bond).
            clause_signals: Clause signals dict.
            link_signals: Linkage signals dict.
            struct_signals: Structure signals dict.
            risk_penalty: Total risk penalty (negative or zero).
            val_composite: Valuation composite 0–1.
            clause_composite: Clause composite 0–1.
            link_composite: Linkage composite 0–1.
            struct_composite: Structure composite 0–1.
            triggered: List of triggered signal names.
            risk_flags: List of risk flag descriptions.

        Returns:
            ScoreResult with full breakdown.
        """
        # Floor each dimension — absence of signals is neutral, not zero
        _val = max(val_composite, self._dimension_floor)
        _clause = max(clause_composite, self._dimension_floor)
        _link = max(link_composite, self._dimension_floor)
        _struct = max(struct_composite, self._dimension_floor)

        base = (
            self._w_val * _val +
            self._w_clause * _clause +
            self._w_link * _link +
            self._w_struct * _struct
        )

        # Convert 0-1 → 0-100
        score_100 = base * 100 + risk_penalty
        score_100 = max(0.0, min(100.0, score_100))

        # Grade
        grade = "C"
        for threshold, g, _ in self.grades:
            if score_100 >= threshold:
                grade = g
                break

        return ScoreResult(
            bond_code=str(row.get("bond_code", row.get("转债代码", ""))),
            bond_name=str(row.get("bond_name", row.get("转债名称", ""))),
            stock_code=str(row.get("stock_code", row.get("正股代码", ""))),
            stock_name=str(row.get("stock_name", row.get("正股名称", ""))),
            cb_price=safe_float(row.get("cb_price", 0), 0.0),
            premium_rate=safe_float(row.get("premium_rate", 0), 0.0),
            conversion_value=safe_float(row.get("conversion_value", 0), 0.0),
            double_low=safe_float(row.get("double_low", 0), 0.0),
            ytm=safe_float(row.get("ytm", 0), 0.0),
            valuation_score=round(_val * 100, 1),
            clause_score=round(_clause * 100, 1),
            linkage_score=round(_link * 100, 1),
            structure_score=round(_struct * 100, 1),
            composite_score=round(score_100, 1),
            risk_penalty=round(risk_penalty, 1),
            grade=grade,
            triggered_signals=triggered,
            risk_flags=risk_flags,
        )

    def rank(
        self,
        results: list[ScoreResult],
        stock_info: pd.DataFrame | None = None,
        industry_neutralize: bool = False,
    ) -> list[ScoreResult]:
        """Sort by composite_score and assign ranks.

        When industry_neutralize is True and stock_info is available,
        z-scores scores within each industry to reduce sector bias.
        """
        if not industry_neutralize or stock_info is None or stock_info.empty:
            sorted_results = sorted(results, key=lambda x: x.composite_score, reverse=True)
            for i, r in enumerate(sorted_results):
                r.rank = i + 1
                r.neutralized_score = r.composite_score
            return sorted_results

        # Build industry lookup: {symbol → industry}
        industry_map: dict[str, str] = {}
        sym_col = next((c for c in ["symbol", "正股代码", "stock_code"] if c in stock_info.columns), None)
        industry_col = next((c for c in ["industry_name", "所属行业", "industry"] if c in stock_info.columns), None)
        if sym_col and industry_col:
            for _, row in stock_info.iterrows():
                sym = str(row.get(sym_col, ""))
                ind = str(row.get(industry_col, ""))
                if sym and ind:
                    industry_map[sym] = ind

        # Group by industry and compute z-scores
        industry_scores: dict[str, list[tuple[int, float]]] = {}
        for idx, r in enumerate(results):
            score = r.composite_score
            ind = industry_map.get(r.stock_code, "__unknown__")
            if ind not in industry_scores:
                industry_scores[ind] = []
            industry_scores[ind].append((idx, score))

        for ind, items in industry_scores.items():
            scores = [s for _, s in items]
            mean = sum(scores) / len(scores)
            if len(scores) < 2:
                std = 1.0
            else:
                variance = sum((s - mean) ** 2 for s in scores) / len(scores)
                std = variance ** 0.5
                if std < 1e-10:
                    std = 1.0
            for idx, score in items:
                results[idx].neutralized_score = round((score - mean) / std * 10 + 50, 1)

        # Sort by neutralized_score descending
        sorted_results = sorted(results, key=lambda x: x.neutralized_score, reverse=True)
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
        return sorted_results

    def grade_description(self, grade: str) -> str:
        """Return Chinese description for a grade."""
        for _, g, desc in self.grades:
            if g == grade:
                return desc
        return "中性"
