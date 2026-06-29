# PulseLab — Single-User Web Performance Lab

A small, local tool that measures **how fast a web page feels for one real visitor**
and explains every number in plain English. Built for someone who has never used
Lighthouse before.

It runs three engines together:

| Engine | What it gives you |
|--------|-------------------|
| 🔦 **Lighthouse** | Google's lab audit — a 0–100 score per category + ranked optimization opportunities. |
| 🎭 **Playwright** | Drives a real browser through multi-step journeys (and logins) so you measure real pages, not just the homepage. |
| 🛠️ **Chrome DevTools (CDP)** | What the live browser actually experienced: FCP / LCP / CLS via PerformanceObserver, Navigation + Resource Timing, console errors, and a screenshot per page. |

No LLM is called. Nothing leaves your machine.

## Folder layout

```
pulselab/
├── server.py            The whole backend (engines + HTTP API). Start here.
├── requirements.txt     Python deps (Flask + Playwright).
├── public/
│   └── index.html       The single-page UI (all tabs).
├── flows/               Saved journeys to audit (.py scripted, or .json page lists).
│   ├── petstore_fish_checkout.py        example: login flow + AUDIT_POINTS
│   └── demo_saucedemo_login.py          example: login + setup() for protected pages
├── reports/             Saved Lighthouse HTML reports (auto-created, gitignored).
├── data/                Saved thresholds + run history (auto-created, gitignored).
└── README.md
```

## Run it

```powershell
# from the pulselab/ folder
py -m pip install -r requirements.txt   # first time only
py -m playwright install chromium       # first time, if no system Chrome/Edge
npm install -g lighthouse               # the Lighthouse CLI must be on PATH
py server.py                            # → http://127.0.0.1:5000/
```

Chrome or Edge must be installed (the CDP engine uses your system browser).
Lighthouse is a Node CLI tool (no Python port exists), so `lighthouse` must be
on your PATH — that is the only Node requirement. Change the port with
`$env:PORT="3000"; py server.py`.

## The full flow

1. **Pick what to measure** — a single URL (Audit tab) or a saved journey (User Journeys tab).
2. **Press Run** — PulseLab opens a real browser, runs the 4-stage pipeline
   (Generate steps → Run engines → Traverse & measure → Build insights).
3. **Overview** — one performance score (0–100) + a plain-English verdict.
4. **Lighthouse tab** — category scores and the opportunities table (ranked by ms saved).
5. **CDP tab** — live FCP/LCP/CLS, the request waterfall, console errors, and a screenshot.
6. **AI Insights tab** — concrete fixes ranked by estimated time saved (rule-based, no LLM).
7. **History & Compare** — every run is logged so you can see if a change helped.

## The metrics, in plain English

- **LCP — Largest Contentful Paint:** when the biggest thing on screen finishes loading. Good ≤ 2.5 s.
- **FCP — First Contentful Paint:** when the first text/image appears. Good ≤ 1.8 s.
- **CLS — Cumulative Layout Shift:** how much the page jumps while loading. Good ≤ 0.1 (0 = nothing moved).
- **TBT — Total Blocking Time:** how long the page was frozen and ignored clicks. Good ≤ 200 ms.
- **SI — Speed Index:** how quickly the page visually fills in. Good ≤ 3.4 s.
- **TTI — Time to Interactive:** when every button reliably responds. Good ≤ 3.8 s.
- **TTFB — Time to First Byte:** how long the server took to start responding. Good ≤ 0.8 s.

Colour bands everywhere: 🟢 **Good** (90–100) · 🟡 **Needs work** (50–89) · 🔴 **Poor** (0–49).
Edit the cutoffs on the **Thresholds** tab — they persist on the server and drive every coloured chip.

## Writing a flow

PulseLab discovers journeys from the `flows/` folder. You can hand-write them, or
describe the journey in plain English and let the **make-flow** Claude agent
generate one (`.claude/agents/make-flow.md`).

**`.py`** — scripted journeys (incl. logins). Define `run(page, context, log)`
and `AUDIT_POINTS`; add an optional `setup(page, context)` for protected pages.
Uses the Playwright **sync** API:

```python
import re

def run(page, context, log):
    log("logging in")
    page.goto("https://example.com/login", wait_until="domcontentloaded")
    page.locator("#user").fill("me")
    page.locator("#pass").fill("secret")
    page.get_by_role("button", name=re.compile("sign in", re.I)).click()
    page.wait_for_load_state("domcontentloaded")

# Optional: re-run before scoring each protected page so it loads logged-in.
def setup(page, context):
    run(page, context, lambda m: None)

AUDIT_POINTS = [
    {"name": "Dashboard", "url": "https://example.com/dashboard"},
    {"name": "Reports",   "url": "https://example.com/reports"},
]
```

**`.json`** — for simple public page lists:

```json
{ "startUrl": "https://example.com", "steps": [ { "name": "Home", "url": "https://example.com" } ] }
```

## HTTP API

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/api/state` | engine status + thresholds + flows + history |
| POST | `/api/audit` | run Lighthouse + CDP on one URL |
| POST | `/api/traverse` | run a flow (Playwright) with per-page Lighthouse + CDP |
| GET/POST | `/api/thresholds` | read / save the SLO bands |
| GET  | `/api/history` | past runs |
| GET  | `/reports` | saved Lighthouse HTML reports |

Single-user: only one audit runs at a time.
