# LMS Model

Premier League **Last Man Standing** pick optimiser for the 2026/27 season.

> 🚧 Being rebuilt from scratch. See [PLAN.md](PLAN.md). The 2025/26 version is preserved at tag `v2025-26`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your ODDS_API_KEY
lms refresh   # ~1 min first time, then seconds
lms status
```
