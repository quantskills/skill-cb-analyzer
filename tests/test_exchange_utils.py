"""Tests for exchange_utils — A-share stock code → exchange suffix mapping."""

import pytest
from core.exchange_utils import resolve_exchange_suffix, EXCHANGE_SUFFIX_MAP


class TestExchangeSuffix:
    """Verify correct exchange suffix for all A-share code prefixes."""

    # Shanghai Main Board
    @pytest.mark.parametrize("code,expected", [
        ("600000", ".SH"), ("601398", ".SH"), ("602000", ".SH"),
        ("603000", ".SH"), ("605000", ".SH"),
    ])
    def test_shanghai_main(self, code, expected):
        assert resolve_exchange_suffix(code).endswith(expected)

    # Shanghai STAR Market
    @pytest.mark.parametrize("code,expected", [
        ("688001", ".SH"), ("688981", ".SH"), ("689009", ".SH"),
    ])
    def test_shanghai_star(self, code, expected):
        assert resolve_exchange_suffix(code).endswith(expected)

    # Shenzhen Main Board
    @pytest.mark.parametrize("code,expected", [
        ("000001", ".SZ"), ("001979", ".SZ"), ("002415", ".SZ"), ("003000", ".SZ"),
    ])
    def test_shenzhen_main(self, code, expected):
        assert resolve_exchange_suffix(code).endswith(expected)

    # Shenzhen ChiNext
    @pytest.mark.parametrize("code,expected", [
        ("300750", ".SZ"), ("300059", ".SZ"), ("301000", ".SZ"),
    ])
    def test_shenzhen_chiNext(self, code, expected):
        assert resolve_exchange_suffix(code).endswith(expected)

    # Beijing Exchange
    @pytest.mark.parametrize("code,expected", [
        ("831445", ".BJ"), ("830799", ".BJ"), ("833171", ".BJ"),
        ("400001", ".BJ"), ("420008", ".BJ"), ("430198", ".BJ"),
    ])
    def test_beijing(self, code, expected):
        assert resolve_exchange_suffix(code).endswith(expected)

    # Unpadded codes
    def test_unpadded_code(self):
        assert resolve_exchange_suffix("1") == "000001.SZ"
        assert resolve_exchange_suffix("688001") == "688001.SH"

    def test_code_with_whitespace(self):
        assert resolve_exchange_suffix(" 688001 ") == "688001.SH"

    # Fallback for truly unknown codes
    def test_fallback_six_prefix(self):
        assert resolve_exchange_suffix("699999").endswith(".SH")

    def test_fallback_five_prefix(self):
        assert resolve_exchange_suffix("500001").endswith(".SZ")

    # Build map sanity
    def test_map_has_entries(self):
        assert len(EXCHANGE_SUFFIX_MAP) >= 15
        assert EXCHANGE_SUFFIX_MAP["688"] == ".SH"
        assert EXCHANGE_SUFFIX_MAP["300"] == ".SZ"
        assert EXCHANGE_SUFFIX_MAP["8"] == ".BJ"
