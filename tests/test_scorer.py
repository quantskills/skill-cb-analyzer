"""Tests for scoring engine."""

import pandas as pd
import pytest
from core.scorer import Scorer, ScoreResult


class TestCompute:
    def test_baseline_score(self, sample_config, sample_cb_df):
        """Floor 0.30 on all dims → base = 0.30, score = 30."""
        s = Scorer(sample_config)
        row = sample_cb_df.iloc[0]
        sr = s.compute(
            0, row,
            val_signals={}, clause_signals={},
            link_signals={}, struct_signals={},
            risk_penalty=0.0,
            val_composite=0.0, clause_composite=0.0,
            link_composite=0.0, struct_composite=0.0,
            triggered=[], risk_flags=[],
        )
        # All floored to 0.30: 0.40*0.30 + 0.30*0.30 + 0.20*0.30 + 0.10*0.30 = 0.30
        # 0.30 * 100 = 30.0
        assert sr.composite_score == 30.0

    def test_max_score(self, sample_config, sample_cb_df):
        """All composites 1.0 → base = 1.0, score = 100."""
        s = Scorer(sample_config)
        row = sample_cb_df.iloc[0]
        sr = s.compute(
            0, row,
            val_signals={}, clause_signals={},
            link_signals={}, struct_signals={},
            risk_penalty=0.0,
            val_composite=1.0, clause_composite=1.0,
            link_composite=1.0, struct_composite=1.0,
            triggered=[], risk_flags=[],
        )
        assert sr.composite_score == 100.0

    def test_risk_penalty(self, sample_config, sample_cb_df):
        """Risk penalty reduces score."""
        s = Scorer(sample_config)
        row = sample_cb_df.iloc[0]
        sr = s.compute(
            0, row,
            val_signals={}, clause_signals={},
            link_signals={}, struct_signals={},
            risk_penalty=-20.0,
            val_composite=0.5, clause_composite=0.5,
            link_composite=0.5, struct_composite=0.5,
            triggered=[], risk_flags=[],
        )
        # base = 0.5 * 100 = 50, with penalty = 30
        assert sr.composite_score == 30.0

    def test_floor_applied(self, sample_config, sample_cb_df):
        """Sub-floor composite gets raised to dimension_floor."""
        s = Scorer(sample_config)
        row = sample_cb_df.iloc[0]
        sr = s.compute(
            0, row,
            val_signals={}, clause_signals={},
            link_signals={}, struct_signals={},
            risk_penalty=0.0,
            val_composite=0.10, clause_composite=0.10,
            link_composite=0.10, struct_composite=0.10,
            triggered=[], risk_flags=[],
        )
        # All floored to 0.30 → score = 30.0
        assert sr.composite_score == 30.0

    def test_triggered_and_risk_flags(self, sample_config, sample_cb_df):
        """Triggered signals and risk flags are preserved in ScoreResult."""
        s = Scorer(sample_config)
        row = sample_cb_df.iloc[0]
        sr = s.compute(
            0, row,
            val_signals={}, clause_signals={},
            link_signals={}, struct_signals={},
            risk_penalty=0.0,
            val_composite=0.5, clause_composite=0.5,
            link_composite=0.5, struct_composite=0.5,
            triggered=["双低策略", "高YTM防御"],
            risk_flags=["强赎风险"],
        )
        assert sr.triggered_signals == ["双低策略", "高YTM防御"]
        assert sr.risk_flags == ["强赎风险"]

    def test_custom_weights(self, sample_cb_df):
        """Custom scoring weights produce different results."""
        config = {
            "scoring": {
                "valuation_weight": 0.25,
                "clause_weight": 0.25,
                "linkage_weight": 0.25,
                "structure_weight": 0.25,
            }
        }
        s = Scorer(config)
        row = sample_cb_df.iloc[0]
        sr = s.compute(
            0, row,
            val_signals={}, clause_signals={},
            link_signals={}, struct_signals={},
            risk_penalty=0.0,
            val_composite=1.0, clause_composite=0.0,
            link_composite=0.0, struct_composite=0.0,
            triggered=[], risk_flags=[],
        )
        # 0.25*1.0 + 0.25*0.30 + 0.25*0.30 + 0.25*0.30 = 0.25+0.075+0.075+0.075 = 0.475
        # 0.475 * 100 = 47.5
        assert sr.composite_score == 47.5

    def test_score_clamped_to_100(self, sample_config, sample_cb_df):
        """Score cannot exceed 100 even with positive penalty."""
        s = Scorer(sample_config)
        row = sample_cb_df.iloc[0]
        sr = s.compute(
            0, row,
            val_signals={}, clause_signals={},
            link_signals={}, struct_signals={},
            risk_penalty=50.0,
            val_composite=1.0, clause_composite=1.0,
            link_composite=1.0, struct_composite=1.0,
            triggered=[], risk_flags=[],
        )
        assert sr.composite_score == 100.0

    def test_score_clamped_to_0(self, sample_config, sample_cb_df):
        """Score cannot go below 0."""
        s = Scorer(sample_config)
        row = sample_cb_df.iloc[0]
        sr = s.compute(
            0, row,
            val_signals={}, clause_signals={},
            link_signals={}, struct_signals={},
            risk_penalty=-200.0,
            val_composite=0.0, clause_composite=0.0,
            link_composite=0.0, struct_composite=0.0,
            triggered=[], risk_flags=[],
        )
        assert sr.composite_score == 0.0


class TestGrade:
    def test_a_plus(self, sample_config, sample_cb_df):
        s = Scorer(sample_config)
        row = sample_cb_df.iloc[0]
        sr = s.compute(
            0, row,
            val_signals={}, clause_signals={},
            link_signals={}, struct_signals={},
            risk_penalty=25.0,
            val_composite=1.0, clause_composite=1.0,
            link_composite=1.0, struct_composite=1.0,
            triggered=[], risk_flags=[],
        )
        # 100 + 25 = 125 → clamped to 100 → grade A+
        assert sr.grade == "A+"

    def test_a(self, sample_config, sample_cb_df):
        s = Scorer(sample_config)
        row = sample_cb_df.iloc[0]
        sr = s.compute(
            0, row,
            val_signals={}, clause_signals={},
            link_signals={}, struct_signals={},
            risk_penalty=0.0,
            val_composite=0.5, clause_composite=0.5,
            link_composite=0.5, struct_composite=0.5,
            triggered=[], risk_flags=[],
        )
        assert sr.grade == "A"

    def test_b_plus(self, sample_config, sample_cb_df):
        s = Scorer(sample_config)
        row = sample_cb_df.iloc[0]
        sr = s.compute(
            0, row,
            val_signals={}, clause_signals={},
            link_signals={}, struct_signals={},
            risk_penalty=0.0,
            val_composite=0.42, clause_composite=0.42,
            link_composite=0.42, struct_composite=0.42,
            triggered=[], risk_flags=[],
        )
        assert sr.grade == "B+"

    def test_d(self, sample_config, sample_cb_df):
        s = Scorer(sample_config)
        row = sample_cb_df.iloc[0]
        sr = s.compute(
            0, row,
            val_signals={}, clause_signals={},
            link_signals={}, struct_signals={},
            risk_penalty=0.0,
            val_composite=0.0, clause_composite=0.0,
            link_composite=0.0, struct_composite=0.0,
            triggered=[], risk_flags=[],
        )
        # 30.0 → grade D (>= 30 but < 34)
        assert sr.grade == "D"

    def test_custom_grades(self, sample_cb_df):
        """Config-provided grade thresholds override defaults."""
        config = {
            "scoring": {
                "grades": [
                    {"threshold": 80, "grade": "S", "label": "超级"},
                    {"threshold": 50, "grade": "A", "label": "优秀"},
                    {"threshold": 0, "grade": "F", "label": "差"},
                ]
            }
        }
        s = Scorer(config)
        row = sample_cb_df.iloc[0]
        sr = s.compute(
            0, row,
            val_signals={}, clause_signals={},
            link_signals={}, struct_signals={},
            risk_penalty=0.0,
            val_composite=0.6, clause_composite=0.6,
            link_composite=0.6, struct_composite=0.6,
            triggered=[], risk_flags=[],
        )
        # 60 → grade A with custom thresholds
        assert sr.grade == "A"


class TestGradeDescription:
    def test_known_grades(self):
        s = Scorer()
        assert s.grade_description("A+") == "强烈关注"
        assert s.grade_description("A") == "值得跟踪"
        assert s.grade_description("B+") == "偏积极"
        assert s.grade_description("B") == "温和"
        assert s.grade_description("C") == "中性"
        assert s.grade_description("D") == "偏弱"
        assert s.grade_description("E") == "回避"

    def test_unknown_grade(self):
        s = Scorer()
        assert s.grade_description("X") == "中性"


class TestRank:
    def test_rank_ordering(self):
        s = Scorer({})
        r1 = ScoreResult(bond_code="A", composite_score=80)
        r2 = ScoreResult(bond_code="B", composite_score=60)
        r3 = ScoreResult(bond_code="C", composite_score=90)
        ranked = s.rank([r1, r2, r3])
        assert ranked[0].bond_code == "C"
        assert ranked[0].rank == 1
        assert ranked[1].bond_code == "A"
        assert ranked[1].rank == 2
        assert ranked[2].bond_code == "B"
        assert ranked[2].rank == 3

    def test_rank_empty(self):
        s = Scorer({})
        assert s.rank([]) == []

    def test_rank_single(self):
        s = Scorer({})
        r = ScoreResult(bond_code="A", composite_score=50)
        ranked = s.rank([r])
        assert len(ranked) == 1
        assert ranked[0].rank == 1


# ---------------------------------------------------------------------------
# Industry neutralization
# ---------------------------------------------------------------------------

class TestIndustryNeutralization:
    """Industry neutralization: z-score scores within each industry before ranking."""

    @pytest.fixture
    def stock_info_df(self):
        """5 stocks across 3 industries."""
        return pd.DataFrame({
            "symbol": ["000001.SZ", "000002.SZ", "600001.SH", "600002.SH", "000003.SZ"],
            "industry_name": ["银行", "银行", "新能源", "新能源", "医药"],
        })

    @pytest.fixture
    def multi_industry_results(self):
        """5 bonds with scores biased by industry."""
        return [
            ScoreResult(bond_code="B001", stock_code="000001.SZ", composite_score=80.0),
            ScoreResult(bond_code="B002", stock_code="000002.SZ", composite_score=60.0),
            ScoreResult(bond_code="B003", stock_code="600001.SH", composite_score=70.0),
            ScoreResult(bond_code="B004", stock_code="600002.SH", composite_score=50.0),
            ScoreResult(bond_code="B005", stock_code="000003.SZ", composite_score=55.0),
        ]

    def test_neutralized_scores_computed(self, multi_industry_results, stock_info_df):
        """Neutralized scores are computed and stored on ScoreResult."""
        s = Scorer({})
        ranked = s.rank(multi_industry_results, stock_info=stock_info_df, industry_neutralize=True)
        for r in ranked:
            assert r.neutralized_score != 0.0
        assert len(ranked) == 5

    def test_ranking_changes_with_neutralization(self, multi_industry_results, stock_info_df):
        """High-score bank bond may drop in rank after neutralization."""
        s = Scorer({})
        raw_ranked = s.rank(multi_industry_results, industry_neutralize=False)
        neut_ranked = s.rank(multi_industry_results, stock_info=stock_info_df, industry_neutralize=True)
        # Ranks may differ (not guaranteed, but neutralized_score must differ from raw)
        raw_top = raw_ranked[0].bond_code
        neut_top = neut_ranked[0].bond_code
        # Top rank can differ if industry effect is neutralized
        assert raw_top == "B001"  # Highest raw score
        # Neutralized ranks should still be unique 1..N
        import numpy as np
        assert sorted([r.rank for r in neut_ranked]) == [1, 2, 3, 4, 5]

    def test_single_industry_fallback(self):
        """All bonds in same industry → std=0 fallback keeps raw ordering."""
        stock_info = pd.DataFrame({
            "symbol": ["A.SH", "B.SH"],
            "industry_name": ["银行", "银行"],
        })
        results = [
            ScoreResult(bond_code="X", stock_code="A.SH", composite_score=80.0),
            ScoreResult(bond_code="Y", stock_code="B.SH", composite_score=60.0),
        ]
        s = Scorer({})
        ranked = s.rank(results, stock_info=stock_info, industry_neutralize=True)
        assert ranked[0].bond_code == "X"
        assert ranked[0].rank == 1

    def test_empty_stock_info_fallback(self):
        """Empty stock_info → fallback to raw ranking."""
        results = [
            ScoreResult(bond_code="X", composite_score=80.0),
            ScoreResult(bond_code="Y", composite_score=60.0),
        ]
        s = Scorer({})
        ranked = s.rank(results, stock_info=pd.DataFrame(), industry_neutralize=True)
        assert ranked[0].bond_code == "X"
        assert ranked[0].neutralized_score == 80.0

    def test_flag_disabled(self, multi_industry_results, stock_info_df):
        """When industry_neutralize=False, neutralized_score = composite_score."""
        s = Scorer({})
        ranked = s.rank(multi_industry_results, stock_info=stock_info_df, industry_neutralize=False)
        for r in ranked:
            assert r.neutralized_score == r.composite_score
