# Data Collection — Getting Started

This guide is written for teammates starting from scratch. Follow every step in order.

---

## 1. Prerequisites

- Python 3.11 or higher
- Git
- Google Chrome

---

## 2. Clone the Repo and Install Dependencies

```bash
git clone <repo-url>
cd Leetcode-Recommendation-System
pip install -e ".[dev]"
pip install curl-cffi   # Cloudflare bypass — not yet on PyPI stable, install separately
```

> If you are using conda, activate your environment first before running these commands.

---

## 3. Create a LeetCode Account

If you do not already have one, sign up at https://leetcode.com. A free account is sufficient.

---

## 4. Get Your Cookies from Chrome

The scraper needs three cookies from your authenticated browser session. LeetCode uses Cloudflare, which requires the `cf_clearance` cookie in addition to the standard session cookies.

**Steps:**

1. Open Chrome and log into https://leetcode.com
2. Navigate to any contest page, e.g. https://leetcode.com/contest/biweekly-contest-181/
3. Wait a moment for the page to fully load (this lets Cloudflare set `cf_clearance`)
4. Press **F12** to open DevTools
5. Go to **Application** → **Storage** → **Cookies** → `https://leetcode.com`
6. Find and copy the **Value** column for each of these three cookies:

| Cookie name | Where to paste it |
|---|---|
| `LEETCODE_SESSION` | `LEETCODE_SESSION=` in `.env.scraper` |
| `csrftoken` | `LEETCODE_CSRF=` in `.env.scraper` |
| `cf_clearance` | `LEETCODE_CF_CLEARANCE=` in `.env.scraper` |

> **Important:** `cf_clearance` expires and is tied to your IP address. If you switch networks or the scraper starts returning 403 errors mid-run, come back here and grab a fresh `cf_clearance` value.

---

## 5. Configure `.env.scraper`

Create the file at the project root (it is already in `.gitignore` — never commit it):

```bash
cp .env.scraper.example .env.scraper   # if an example exists
# or just create it manually
```

Open `.env.scraper` and fill in your three cookie values:

```
LEETCODE_SESSION=<paste your value here>
LEETCODE_CSRF=<paste your value here>
LEETCODE_CF_CLEARANCE=<paste your value here>
```

---

## 6. Run the Scraper

Run from the **project root** (not from inside `scripts/`). Use the range assigned to your machine:

| Machine | Command |
|---|---|
| Machine 1 | `python -m scripts.data_collection.run --from 180 --to 121` |
| Machine 2 | `python -m scripts.data_collection.run --from 120 --to 61` |
| Machine 3 | `python -m scripts.data_collection.run --from 60 --to 0` |

You will see per-page progress in the terminal. Each contest auto-detects its own page count.

---

## 7. Resuming an Interrupted Run

If the scraper stops for any reason (network error, expired cookie, etc.), just re-run the exact same command. Already-scraped pages are detected from the output files and skipped automatically.

If you need to re-scrape a range from scratch:

```bash
python -m scripts.data_collection.run --from 180 --to 121 --force
```

---

## 8. Output

Each contest is saved as a separate JSONL file (one contestant entry per line):

```
data/raw/contests/
├── biweekly-contest-180.jsonl
├── biweekly-contest-179.jsonl
└── ...
```

To load all contests into a single DataFrame for analysis:

```python
import pandas as pd
from pathlib import Path

df = pd.concat(
    pd.read_json(f, lines=True)
    for f in sorted(Path("data/raw/contests").glob("*.jsonl"))
)
```

---

## 9. Sharing Data with the Team

Once your range is complete, share the `.jsonl` files with your teammates via the agreed method (shared drive, S3, etc.). Do **not** commit data files to git — `data/raw/` is in `.gitignore`.
