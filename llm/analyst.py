"""LLM analyst: per-bond analysis via DeepSeek/Claude API.

Analyzes top-ranked convertible bonds individually, covering:
  - Technical view (valuation + clause signals)
  - Risk assessment (credit + redemption risk)
  - Strategy suggestion

Supports Anthropic Claude and DeepSeek (via Anthropic-compatible endpoint).
Set ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic for DeepSeek.

Backend: Pluggable LLM backends via LLMBackend Protocol. Default is
AnthropicBackend (Anthropic-compatible API). Set llm.provider in config
to swap backends.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for pluggable LLM backends."""

    def generate(self, prompt: str, model: str, max_tokens: int, **kwargs) -> str:
        """Generate text from a prompt. Returns empty string on failure."""
        ...


class AnthropicBackend:
    """LLM backend using Anthropic-compatible API (Claude, DeepSeek, etc.)."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._client = None
        self._api_key = os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY") or ""
        self._base_url = os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        self._timeout = float(cfg.get("timeout", 120.0))
        self._max_retries = int(cfg.get("max_retries", 2))

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            return None
        try:
            from anthropic import Anthropic
            self._client = Anthropic(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        except Exception as e:
            logger.warning("Failed to init Anthropic client: %s", e)
            self._client = None
        return self._client

    def generate(self, prompt: str, model: str, max_tokens: int, **kwargs) -> str:
        client = self._get_client()
        if client is None:
            return ""
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # Warn if response was truncated due to max_tokens
        if hasattr(resp, "stop_reason") and resp.stop_reason == "max_tokens":
            logger.warning("LLM response truncated (max_tokens reached)")
        content = resp.content
        if isinstance(content, list):
            text_blocks = [b.text for b in content if hasattr(b, "text")]
            return "\n".join(text_blocks)
        elif hasattr(content, "text"):
            return content.text
        return str(content)

DEFAULT_CB_PROMPT_TEMPLATE = """你是一位资深可转债分析师，专精A股转债的估值、条款博弈和正股联动分析。
请基于以下数据，对该可转债做一次专业的逐券分析。

## 转债基本信息
- 转债名称：{bond_name}（{bond_code}）
- 正股：{stock_name}（{stock_code}）
- 转债价格：{cb_price:.2f} 元
- 转股溢价率：{premium_rate:.1f}%
- 转股价值：{conversion_value:.2f}
- 双低值：{double_low:.1f}
- 到期收益率：{ytm:.2f}%

## 信号检测结果
{signal_summary}

## 评分
- 估值维度：{val_score:.0f}/100
- 条款维度：{clause_score:.0f}/100
- 正股联动：{link_score:.0f}/100
- 市场结构：{struct_score:.0f}/100
- 综合评分：{composite:.0f}/100（{grade}）

## 风险提示
{risk_summary}

## 要求
1. **一句话定性**：这只转债当前处于什么状态？（进攻/防守/博弈/回避）
2. **估值分析**：当前价格和溢价率是否合理？双低值有吸引力吗？
3. **条款研判**：是否有强赎风险、下修可能、或回售保护？
4. **正股联动**：正股走势对转债的影响方向。
5. **综合建议**：对这种转债，应该关注什么？用什么策略？（注意用「值得关注」「可跟踪」等措辞，禁止推荐买入/卖出/目标价）
6. 控制在 200-350 字。"""


class CBAnalyst:
    """LLM analyst for individual convertible bonds.

    Args:
        config: Optional config dict with 'llm' section.
        backend: Optional LLMBackend instance (overrides config provider).
    """

    def __init__(self, config: Optional[dict] = None,
                 backend: LLMBackend | None = None):
        cfg = config or {}
        llm_cfg = cfg.get("llm", {})
        self._model = os.getenv("ANTHROPIC_MODEL") or llm_cfg.get("model", "deepseek-v4-pro")
        self._max_tokens = int(llm_cfg.get("max_tokens", 2048))
        self._llm_retries = int(llm_cfg.get("llm_retries", 3))
        self._prompt_template = llm_cfg.get("prompt_template") or DEFAULT_CB_PROMPT_TEMPLATE

        # Pluggable backend
        if backend is not None:
            self._backend: LLMBackend | None = backend
        else:
            provider = llm_cfg.get("provider", "anthropic")
            if provider == "anthropic":
                self._backend = AnthropicBackend(llm_cfg)
            else:
                logger.warning("Unknown LLM provider '%s', using fallback", provider)
                self._backend = None

    def analyze(self, bond_data: dict) -> str:
        """Run LLM analysis for a single CB.

        Args:
            bond_data: Dict with keys matching prompt template placeholders.

        Returns:
            Analysis text, or fallback analysis on failure.
        """
        if self._backend is None:
            return self._fallback_analysis(bond_data)

        prompt = self._prompt_template.format(**bond_data)

        for attempt in range(1, self._llm_retries + 1):
            try:
                text = self._backend.generate(prompt, self._model, self._max_tokens)
                if text and text.strip():
                    return text.strip()
                logger.warning("LLM returned empty text for %s", bond_data.get("bond_name", "?"))
            except Exception as e:
                logger.warning("LLM attempt %d/%d failed for %s: %s",
                               attempt, self._llm_retries, bond_data.get("bond_name", "?"), e)
                if attempt < self._llm_retries:
                    time.sleep(2 ** attempt)

        return self._fallback_analysis(bond_data)

    def _fallback_analysis(self, bond_data: dict) -> str:
        """Generate a rule-based analysis when LLM is unavailable."""
        name = bond_data.get("bond_name", "未知")
        price = bond_data.get("cb_price", 0)
        premium = bond_data.get("premium_rate", 0)
        grade = bond_data.get("grade", "C")
        dl = bond_data.get("double_low", 0)
        ytm = bond_data.get("ytm", 0)

        parts = [f"{name}："]
        if grade in ("A+", "A"):
            parts.append("综合信号偏积极。")
        elif grade in ("B+", "B"):
            parts.append("信号温和偏多，值得跟踪。")
        elif grade == "C":
            parts.append("多空信号均衡，观望为主。")
        else:
            parts.append("信号偏弱，注意风险。")

        if premium < 0:
            parts.append(f"当前折价{abs(premium):.1f}%，有转股套利空间但需关注正股流动性。")
        elif premium < 20:
            parts.append(f"溢价率{premium:.1f}%处于合理区间。")
        elif premium < 50:
            parts.append(f"溢价率{premium:.1f}%偏高，进攻性受限。")
        else:
            parts.append(f"溢价率{premium:.1f}%极高，转债弹性差。")

        if dl > 0 and dl < 130:
            parts.append(f"双低值{dl:.1f}具备一定性价比。")

        if isinstance(ytm, float) and ytm > 3:
            parts.append(f"YTM={ytm:.1f}%提供债底保护。")

        if price < 100:
            parts.append("转债破发，关注回售和下修博弈机会。")

        return "".join(parts)

    def analyze_batch(
        self, bonds: list[dict], delay: float = 0.5
    ) -> dict[str, str]:
        """Analyze multiple bonds sequentially.

        Args:
            bonds: List of bond data dicts.
            delay: Seconds between API calls.

        Returns:
            Dict mapping bond_code → analysis text.
        """
        results = {}
        for i, bond in enumerate(bonds):
            logger.info("LLM analysis %d/%d: %s", i + 1, len(bonds), bond.get("bond_name", "?"))
            results[bond.get("bond_code", str(i))] = self.analyze(bond)
            if i < len(bonds) - 1:
                time.sleep(delay)
        return results
