# Portable Loader Prompt

Use this prompt in agents that do not natively discover `SKILL.md` folders.

```text
You have access to a local skill named skill-cb-analyzer at:
<CB_ANALYZER_SKILL_ROOT>

When the user request matches this skill's SKILL.md description:
1. Read <CB_ANALYZER_SKILL_ROOT>/SKILL.md.
2. Follow the workflow and guardrails in that file exactly.
3. Load referenced files under <CB_ANALYZER_SKILL_ROOT>/references/ only when needed.
4. Run bundled scripts from the skill root only after reading the relevant instructions.
5. Preserve documented API names, parameters, file paths, formulas, validation limits, and freshness notes.
6. Do not invent data interfaces, credentials, factor definitions, or runtime behavior that is not supported by the skill files.
```

## MCP Server Integration

The skill can also be called by LLMs via its MCP server. Start the server with:

```bash
cd <SKILL_ROOT>
python mcp_server.py
```

### Claude Code (claude.ai/code)

Add to `.claude/mcp.json` or `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "cb-analyzer": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "<SKILL_ROOT>",
      "env": {
        "ANTHROPIC_AUTH_TOKEN": "sk-xxx",
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
        "ANTHROPIC_MODEL": "claude-sonnet-4-20250514"
      }
    }
  }
}
```

### Available Tools

| Tool | Description |
|---|---|
| `run_cb_analyzer` | Run full daily CB analysis with optional LLM per-bond analysis |
| `get_latest_report` | Read the most recent CB report (full content or preview) |
| `check_trading_day` | Verify if a date is an A-share trading day |
| `search_bonds` | Search ranked bonds by name/code in the latest report |

### Direct CLI

```bash
python run.py                       # Latest trading day (LLM enabled by default)
python run.py --date 20260701       # Specify date
python run.py --no-llm              # Skip LLM, rule-engine only
python run.py --top-n 30 --verbose  # Custom params
python run.py --backtest            # Enable backtest analysis
```

All data is from real sources (AKShare + Pandadata + LLM API). No mock/dry-run mode.
