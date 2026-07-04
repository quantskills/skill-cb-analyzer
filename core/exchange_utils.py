"""Exchange suffix mapping for A-share stock codes.

Centralises exchange resolution so stock codes are correctly mapped to
Shanghai (.SH), Shenzhen (.SZ), or Beijing (.BJ) suffixes.

Rules (current as of 2026):
  Shanghai Main Board:  600xxx, 601xxx, 602xxx, 603xxx, 605xxx → .SH
  Shanghai STAR Market: 688xxx, 689xxx → .SH
  Shenzhen Main Board:  000xxx, 001xxx, 002xxx, 003xxx → .SZ
  Shenzhen ChiNext:     300xxx, 301xxx → .SZ
  Beijing Exchange:     400xxx, 420xxx, 430xxx, 8xxxxx → .BJ

Unknown codes fall back to the legacy heuristic (6-prefix → .SH, else .SZ).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Build the prefix → suffix lookup table
# ---------------------------------------------------------------------------

EXCHANGE_SUFFIX_MAP: dict[str, str] = {}


def _build_map() -> None:
    """Populate EXCHANGE_SUFFIX_MAP with known A-share code prefixes."""
    # Shanghai Main Board
    for prefix in ("600", "601", "602", "603", "605"):
        EXCHANGE_SUFFIX_MAP[prefix] = ".SH"
    # Shanghai STAR Market (科创板)
    for prefix in ("688", "689"):
        EXCHANGE_SUFFIX_MAP[prefix] = ".SH"
    # Shenzhen Main Board
    for prefix in ("000", "001", "002", "003"):
        EXCHANGE_SUFFIX_MAP[prefix] = ".SZ"
    # Shenzhen ChiNext (创业板)
    for prefix in ("300", "301"):
        EXCHANGE_SUFFIX_MAP[prefix] = ".SZ"
    # Beijing Exchange (北交所) — 4xx 退市板 + 8xxxxx
    for prefix in ("400", "420", "430", "8"):
        EXCHANGE_SUFFIX_MAP[prefix] = ".BJ"


_build_map()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_exchange_suffix(stock_code: str) -> str:
    """Return *stock_code* with the correct A-share exchange suffix.

    Args:
        stock_code: Raw stock code, e.g. ``"688001"``, ``"300750"``, ``"831445"``.

    Returns:
        Code with exchange suffix, e.g. ``"688001.SH"``.

    If the prefix is unrecognised the function falls back to the legacy
    heuristic: codes starting with ``"6"`` → ``.SH``, everything else → ``.SZ``.
    """
    code = str(stock_code).strip().zfill(6)

    # Try 3-digit prefix match first (covers 600-605, 688-689, 000-003, 300-301, 400-430)
    prefix_3 = code[:3]
    if prefix_3 in EXCHANGE_SUFFIX_MAP:
        return f"{code}{EXCHANGE_SUFFIX_MAP[prefix_3]}"

    # Try 1-digit prefix (covers 8xxxxx → .BJ)
    prefix_1 = code[0]
    if prefix_1 in EXCHANGE_SUFFIX_MAP:
        return f"{code}{EXCHANGE_SUFFIX_MAP[prefix_1]}"

    # Legacy fallback for unknown codes
    return f"{code}.SH" if code.startswith("6") else f"{code}.SZ"
