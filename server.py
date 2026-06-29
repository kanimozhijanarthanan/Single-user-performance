"""
=============================================================================
 PulseLab — Single-User Web Performance Lab  (Python / Flask port)
=============================================================================
 A local tool that measures how fast a web page feels for ONE real visitor
 and explains every number in plain English.

 Three engines:
   - Lighthouse   : Google's lab audit (0-100 scores + opportunities)
                    Called via the Lighthouse CLI (no Python Lighthouse lib
                    exists), through a subprocess.
   - Playwright   : drives a real browser through multi-step journeys/logins
                    (playwright-python, sync API).
   - Chrome (CDP) : live FCP/LCP/CLS, timing, request waterfall, screenshot.

 No LLM. Runs entirely offline. The UI lives in public/index.html.

 HTTP API
   GET  /                 the single-page app
   GET  /api/state        engine status + thresholds + flows + history
   POST /api/audit        audit one URL          { url, device, throttling }
   POST /api/traverse     audit a saved flow     { flow, device }
   GET  /api/thresholds   read the SLO bands
   POST /api/thresholds   save the SLO bands     { thresholds }
   GET  /api/history      past runs
   GET  /reports          saved Lighthouse HTML reports

 Run it:
   pip install -r requirements.txt
   python -m playwright install chromium      # first time, if no system browser
   npm install -g lighthouse                  # the Lighthouse CLI must be on PATH
   python server.py                           # -> http://127.0.0.1:5000/
=============================================================================
"""

import ast
import base64
import html as html_lib
import importlib.util
import json as json_lib
import logging
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_file, send_from_directory
from playwright.sync_api import sync_playwright

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
FLOWS_DIR = os.path.join(BASE_DIR, "flows")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR = os.path.join(BASE_DIR, "data")
PROFILES_DIR = os.path.join(DATA_DIR, "profiles")  # saved Playwright storage_state profiles
THRESHOLDS_FILE = os.path.join(DATA_DIR, "thresholds.json")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")

for _dir in (REPORTS_DIR, DATA_DIR, PROFILES_DIR):
    os.makedirs(_dir, exist_ok=True)


def read_json(file, fallback):
    try:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as fh:
                return json_lib.load(fh)
    except Exception:
        pass
    return fallback


def write_json(file, obj):
    with open(file, "w", encoding="utf-8") as fh:
        json_lib.dump(obj, fh, indent=2)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
#  Optional LLM (Azure OpenAI) — used for NLP->flow generation and AI Analysis.
#  Entirely optional: with no key configured the app falls back to the existing
#  rule-based behaviour. The key is read ONLY from the environment / .env and is
#  NEVER logged, NEVER returned to the browser.
# =============================================================================
def _load_dotenv(path):
    """Minimal .env loader (no python-dotenv dependency). KEY=VALUE per line."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:   # real env vars win over .env
                    os.environ[k] = v
    except Exception:
        pass


_load_dotenv(os.path.join(BASE_DIR, ".env"))


def llm_config():
    """Azure OpenAI settings from env. Returns a dict; values may be empty."""
    return {
        "endpoint": (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/"),
        "key": os.getenv("AZURE_OPENAI_KEY") or "",
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT") or "",
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview",
    }


def llm_enabled():
    c = llm_config()
    return bool(c["endpoint"] and c["key"] and c["deployment"])


# AI insights are OPT-IN per run: the audit/traverse request sets this from its
# `ai` flag. When False, no LLM is called and the report shows data only (the
# deterministic fallback), even if a key is configured.
_WANT_AI = False


def ai_active():
    """True only when the user asked for AI on this run AND a key is configured."""
    return bool(_WANT_AI and llm_enabled())


def llm_chat(system, user, max_tokens=1200, temperature=0.2):
    """Single chat-completion call to Azure OpenAI via urllib (no SDK dependency).
    Returns the assistant text. Raises on any failure (callers catch + fall back).
    The api-key is sent only in the request header and is never logged."""
    import urllib.request
    import urllib.error

    c = llm_config()
    if not (c["endpoint"] and c["key"] and c["deployment"]):
        raise RuntimeError("LLM not configured")
    url = (f"{c['endpoint']}/openai/deployments/{c['deployment']}"
           f"/chat/completions?api-version={c['api_version']}")
    payload = json_lib.dumps({
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("api-key", c["key"])
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json_lib.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Surface status only — never the key or full request.
        raise RuntimeError(f"Azure OpenAI HTTP {e.code}")
    except Exception as e:
        raise RuntimeError(f"Azure OpenAI call failed: {type(e).__name__}")
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError("Azure OpenAI returned no content")


# One-time migration: older history entries were saved before runs had a stable
# `id`. Backfill ids so those runs can be deleted/compared from the UI too.
def _backfill_history_ids():
    try:
        if not os.path.exists(HISTORY_FILE):
            return
        history = read_json(HISTORY_FILE, [])
        changed = False
        for run in history:
            if not run.get("id"):
                run["id"] = f"{int(time.time() * 1000):x}-{uuid.uuid4().hex[:6]}"
                changed = True
        if changed:
            write_json(HISTORY_FILE, history)
    except Exception:
        pass  # non-fatal


_backfill_history_ids()

# single-user lock: only one audit/traversal at a time
audit_running = False
# Live progress for the running flow — polled by the Audit page to stream the
# browser view + step log while the journey runs.
LIVE = {"running": False, "flow": None, "steps": [], "current": None, "shot": None}


def live_reset(flow):
    global LIVE
    LIVE = {"running": True, "flow": flow, "steps": [], "current": None, "shot": None}


def live_step(name, url, shot=None):
    LIVE["current"] = name
    LIVE["steps"].append({"name": name, "url": url, "at": int(time.time() * 1000)})
    if shot:
        LIVE["shot"] = shot


# =============================================================================
#  THRESHOLDS (SLOs)
#  The single source of truth for the Good / Needs-work / Poor colour bands
#  shown across the UI. Defaults are the standard Web Vitals + Chrome cutoffs.
# =============================================================================
DEFAULT_THRESHOLDS = {
    "score":         {"good": 90,   "ni": 50,   "unit": "",   "label": "Lighthouse perf score",          "note": "Higher is better; above Good = green."},
    "lcp":           {"good": 2500, "ni": 4000, "unit": "ms", "label": "Largest Contentful Paint (LCP)",  "note": "Largest above-the-fold paint."},
    "fcp":           {"good": 1800, "ni": 3000, "unit": "ms", "label": "First Contentful Paint (FCP)",    "note": "Time to first non-blank pixel."},
    "cls":           {"good": 0.1,  "ni": 0.25, "unit": "",   "label": "Cumulative Layout Shift (CLS)",   "note": "Unitless. Lower is better."},
    "tbt":           {"good": 200,  "ni": 600,  "unit": "ms", "label": "Total Blocking Time (TBT)",       "note": "Main-thread blocking after FCP."},
    "ttfb":          {"good": 800,  "ni": 1800, "unit": "ms", "label": "Time to First Byte (TTFB)",       "note": "Server response latency."},
    "pageLoad":      {"good": 3400, "ni": 5800, "unit": "ms", "label": "Page load (DCL to load)",         "note": "Document load wall-clock."},
    "requests":      {"good": 50,   "ni": 100,  "unit": "",   "label": "Total network requests",          "note": "Per page."},
    "transferKb":    {"good": 1500, "ni": 3000, "unit": "KB", "label": "Bytes transferred",               "note": "Payload across all requests."},
    "consoleErrors": {"good": 0,    "ni": 3,    "unit": "",   "label": "Console errors",                  "note": "Good = 0; Poor = 3 or more."},
}


def read_thresholds():
    saved = read_json(THRESHOLDS_FILE, None)
    if not saved:
        return {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}
    # Merge so newly added default keys still appear on top of an old saved file.
    merged = {}
    for key, default in DEFAULT_THRESHOLDS.items():
        merged[key] = {**default, **(saved.get(key) or {})}
    return merged


# =============================================================================
#  METRIC DICTIONARY
#  Plain-English explanation of each metric (shown on hover in the UI), plus
#  the mapping from our short keys to Lighthouse's audit ids.
# =============================================================================
METRICS = {
    "lcp":  {"label": "LCP",  "lhId": "largest-contentful-paint", "what": "Time until the biggest thing on screen (hero image / headline) is fully shown.",   "why": 'The moment the page looks "ready" — the headline speed number.'},
    "fcp":  {"label": "FCP",  "lhId": "first-contentful-paint",   "what": "Time until the first piece of content appears.",                                    "why": "A blank screen feels broken; first paint shows it is working."},
    "cls":  {"label": "CLS",  "lhId": "cumulative-layout-shift",  "what": "How much the page jumps around while loading.",                                     "why": "Unexpected movement causes mis-taps. 0 = nothing moved."},
    "tbt":  {"label": "TBT",  "lhId": "total-blocking-time",      "what": "How long the page was frozen and could not respond to clicks while loading code.",  "why": "A page can look ready but ignore clicks — feels unresponsive."},
    "si":   {"label": "SI",   "lhId": "speed-index",              "what": "How quickly the page visually fills in from blank to complete.",                    "why": 'Captures the overall "it loaded fast" feeling.'},
    "tti":  {"label": "TTI",  "lhId": "interactive",              "what": "Time until the page is fully usable — every button reliably responds.",             "why": "When the user can actually start doing things."},
    "ttfb": {"label": "TTFB", "lhId": "server-response-time",     "what": "Time for the server to send the first byte of the page.",                           "why": "High TTFB = a slow backend before the browser can even start."},
}

LH_CATEGORIES = ["performance", "accessibility", "best-practices", "seo"]


# =============================================================================
#  SCORING HELPERS
# =============================================================================
def score_band(score):
    """A 0..1 Lighthouse score -> a band level used for colour + label."""
    if score is None:
        return "na"
    if score >= 0.9:
        return "good"
    if score >= 0.5:
        return "average"
    return "poor"


def value_band(key, value, thresholds):
    """A raw metric value vs. its threshold -> band level. Lower is better for
    every metric except `score`, which is higher-is-better."""
    t = thresholds.get(key)
    if not t or value is None:
        return "na"
    if key == "score":
        return "good" if value >= t["good"] else "average" if value >= t["ni"] else "poor"
    return "good" if value <= t["good"] else "average" if value <= t["ni"] else "poor"


def verdict(perf_score):
    if perf_score is None:
        return "We could not measure performance for this page."
    if perf_score >= 90:
        return "Fast for a single visitor. Most users will have a smooth experience."
    if perf_score >= 50:
        return "Okay but has room to improve. Some users will notice waiting."
    return "Slow for a single visitor. Users are likely to wait noticeably and may leave."


def build_insights(metrics, lhr):
    """Rule-based 'insights' (no LLM): concrete fixes derived from the real audit."""
    tips = []
    by_key = {m["key"]: m for m in metrics}

    def bad(k):
        return k in by_key and by_key[k]["level"] != "good"

    if bad("lcp"):
        tips.append("LCP is high — the main image/headline takes too long. Compress the hero image, preload it, and avoid lazy-loading above the fold.")
    if bad("tbt"):
        tips.append("TBT is high — too much JavaScript blocks the main thread. Split bundles, defer non-critical scripts, and remove unused code.")
    if bad("cls"):
        tips.append("CLS is high — content shifts as it loads. Set explicit width/height on images and reserve space for ads/embeds.")
    if bad("fcp"):
        tips.append("FCP is slow — first paint is delayed. Reduce render-blocking CSS/JS and improve server response (TTFB).")

    # Top Lighthouse opportunities, by estimated time saved.
    opps = [
        a for a in lhr.get("audits", {}).values()
        if (a.get("details") or {}).get("type") == "opportunity"
        and (a["details"].get("overallSavingsMs") or 0) > 0
    ]
    opps.sort(key=lambda a: a["details"]["overallSavingsMs"], reverse=True)
    for a in opps[:3]:
        tips.append(f"{a['title']} (save ~{round(a['details']['overallSavingsMs'])} ms)")

    if not tips:
        tips.append("No significant performance problems found for a single visitor. Good job!")
    return tips


def _analysis_digest(name, url, score, verdict_text, metrics, opportunities, cdp):
    """Compact, LLM-friendly summary of one page's real audit data."""
    rb = (cdp or {}).get("responseBreakdown") or {}
    slow = (cdp or {}).get("slowestRequest") or {}
    return {
        "page": name, "url": url,
        "performanceScore": score, "verdict": verdict_text,
        "metrics": [{"k": m["label"], "value": m["value"], "band": m["level"]} for m in (metrics or [])],
        "topOpportunities": [{"title": o["title"], "saveMs": o.get("saveMs")} for o in (opportunities or [])[:5]],
        "cdp": {
            "perceivedMs": rb.get("perceivedMs"), "serverMs": rb.get("serverMs"), "renderMs": rb.get("renderMs"),
            "totalRequests": (cdp or {}).get("totalRequests"), "transferKb": (cdp or {}).get("transferKb"),
            "consoleErrors": (cdp or {}).get("consoleErrors"),
            "slowestRequest": {"name": slow.get("name"), "ms": slow.get("durationMs"), "type": slow.get("type")} if slow else None,
        },
    }


def _llm_analysis_from_digest(digest):
    """Run the LLM over a page digest and return a structured analysis dict, or
    None on any failure. Used by the on-demand /api/page-insights endpoint, so it
    does NOT consult the per-run opt-in — it only requires a configured key."""
    if not llm_enabled():
        return None
    system = (
        "You are a web-performance analyst for a SINGLE-USER performance lab. "
        "You are given real audit data for one web page (Lighthouse metrics + live Chrome "
        "measurements). The key idea: the response time a real user FEELS is the server "
        "response (TTFB) PLUS browser rendering time — so a page whose server replies in 5s "
        "but renders in 5s more is really a 10s experience. "
        "Explain the page's performance for a NON-TECHNICAL third person, then give concrete "
        "fixes. Respond with ONLY valid JSON, no markdown, in this exact shape: "
        '{"summary": "one sentence", "plainEnglish": "2-3 sentences a non-technical person '
        'understands, mentioning the server-vs-render split and total perceived wait", '
        '"topIssues": ["..."], "recommendations": ["concrete fix", "..."]}'
    )
    user = "Audit data (JSON):\n" + json_lib.dumps(digest, indent=2)
    try:
        raw = llm_chat(system, user, max_tokens=900, temperature=0.2)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        obj = json_lib.loads(raw)
        if isinstance(obj, dict) and obj.get("plainEnglish"):
            obj["_engine"] = "azure-openai"
            return obj
    except Exception:
        pass
    return None


def build_llm_analysis(name, url, score, verdict_text, metrics, opportunities, cdp):
    """LLM-written analysis from the real audit data. Returns a structured dict, or
    None when AI is off for this run (callers keep the rule-based insights)."""
    if not ai_active():
        return None
    digest = _analysis_digest(name, url, score, verdict_text, metrics, opportunities, cdp)
    system = (
        "You are a web-performance analyst for a SINGLE-USER performance lab. "
        "You are given real audit data for one web page (Lighthouse metrics + live Chrome "
        "measurements). The key idea: the response time a real user FEELS is the server "
        "response (TTFB) PLUS browser rendering time — so a page whose server replies in 5s "
        "but renders in 5s more is really a 10s experience. "
        "Explain the page's performance for a NON-TECHNICAL third person, then give concrete "
        "fixes. Respond with ONLY valid JSON, no markdown, in this exact shape: "
        '{"summary": "one sentence", "plainEnglish": "2-3 sentences a non-technical person '
        'understands, mentioning the server-vs-render split and total perceived wait", '
        '"topIssues": ["..."], "recommendations": ["concrete fix", "..."]}'
    )
    user = "Audit data (JSON):\n" + json_lib.dumps(digest, indent=2)
    try:
        raw = llm_chat(system, user, max_tokens=900, temperature=0.2)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        obj = json_lib.loads(raw)
        if isinstance(obj, dict) and obj.get("plainEnglish"):
            obj["_engine"] = "azure-openai"
            return obj
    except Exception:
        pass
    return None


def extract_metrics(lhr):
    """Explained core metrics extracted from a Lighthouse result."""
    out = []
    audits = lhr.get("audits", {})
    for key, info in METRICS.items():
        audit = audits.get(info["lhId"])
        if not audit:
            continue
        score = audit.get("score")
        out.append({
            "key": key,
            "label": info["label"],
            "value": audit.get("displayValue") or "n/a",
            "numeric": audit.get("numericValue"),
            "score": None if score is None else round(score * 100),
            "level": score_band(score),
            "what": info["what"],
            "why": info["why"],
        })
    return out


def extract_categories(lhr):
    cats = lhr.get("categories", {}) or {}
    out = []
    for k in LH_CATEGORIES:
        if k in cats:
            score = cats[k].get("score") or 0
            out.append({
                "key": k,
                "label": cats[k].get("title"),
                "score": round(score * 100),
                "level": score_band(cats[k].get("score")),
            })
    return out


def audit_category(lhr, audit_id):
    """Which Lighthouse category each audit belongs to (for the table's Category col)."""
    for key, cat in (lhr.get("categories", {}) or {}).items():
        if any(r.get("id") == audit_id for r in (cat.get("auditRefs") or [])):
            return cat.get("title") or key
    return ""


def extract_opportunities(lhr):
    """Failing audits / opportunities for the Lighthouse tab table."""
    rows = []
    for audit_id, a in lhr.get("audits", {}).items():
        details = a.get("details") or {}
        is_opp = details.get("type") == "opportunity"
        score = a.get("score")
        failed = (
            score is not None and score < 0.9
            and a.get("scoreDisplayMode") not in ("informative", "notApplicable")
        )
        if not (is_opp or failed):
            continue
        desc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", a.get("description") or "")[:220]
        rows.append({
            "id": audit_id,
            "title": a.get("title"),
            "description": desc,
            "saveMs": round(details.get("overallSavingsMs") or 0) if is_opp else None,
            "saveBytes": round(details["overallSavingsBytes"]) if details.get("overallSavingsBytes") else None,
            "score": "FAIL" if (score == 0 or score is None) else round(score * 100),
            "category": audit_category(lhr, audit_id),
        })
    rows.sort(key=lambda r: r["saveMs"] or 0, reverse=True)
    return rows[:30]


# =============================================================================
#  ENGINE: Lighthouse (lab scoring) — via the Lighthouse CLI subprocess.
#  There is no Python Lighthouse library; we shell out to the `lighthouse` CLI
#  (npm i -g lighthouse) pointed at the same debug-port Chrome we control.
# =============================================================================
def _lighthouse_cli():
    """Locate the Lighthouse CLI. Prefer PATH, but fall back to the known npm
    global install locations — the server is often launched from an IDE/process
    whose environment doesn't inherit the user's updated PATH, so PATH alone
    can miss a perfectly-installed CLI."""
    found = shutil.which("lighthouse") or shutil.which("lighthouse.cmd")
    if found:
        return found
    # Fallbacks: npm's global bin folders on Windows / macOS / Linux.
    candidates = []
    appdata = os.environ.get("APPDATA")
    if appdata:  # Windows: %APPDATA%\npm\lighthouse.cmd
        candidates += [os.path.join(appdata, "npm", "lighthouse.cmd"),
                       os.path.join(appdata, "npm", "lighthouse")]
    home = os.path.expanduser("~")
    candidates += [
        os.path.join(home, "AppData", "Roaming", "npm", "lighthouse.cmd"),
        "/usr/local/bin/lighthouse", "/usr/bin/lighthouse",
        os.path.join(home, ".npm-global", "bin", "lighthouse"),
    ]
    # Last resort: ask npm itself where its global prefix is.
    try:
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if npm:
            prefix = subprocess.run([npm, "prefix", "-g"], capture_output=True, text=True, timeout=15).stdout.strip()
            if prefix:
                candidates += [os.path.join(prefix, "lighthouse.cmd"), os.path.join(prefix, "lighthouse"),
                               os.path.join(prefix, "bin", "lighthouse")]
    except Exception:
        pass
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _lighthouse_module_dir():
    """Locate the installed `lighthouse` MODULE directory (the folder that holds
    core/index.js), so the Node-API runner can import it. Derived from the CLI
    shim's location or npm's global prefix."""
    candidates = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(os.path.join(appdata, "npm", "node_modules", "lighthouse"))
    home = os.path.expanduser("~")
    candidates += [
        os.path.join(home, "AppData", "Roaming", "npm", "node_modules", "lighthouse"),
        "/usr/local/lib/node_modules/lighthouse",
        "/usr/lib/node_modules/lighthouse",
        os.path.join(home, ".npm-global", "lib", "node_modules", "lighthouse"),
    ]
    try:
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if npm:
            prefix = subprocess.run([npm, "prefix", "-g"], capture_output=True,
                                    text=True, timeout=15).stdout.strip()
            if prefix:
                candidates += [os.path.join(prefix, "node_modules", "lighthouse"),
                               os.path.join(prefix, "lib", "node_modules", "lighthouse")]
    except Exception:
        pass
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "core", "index.js")):
            return c
    return None


def _node_bin():
    """Locate Node.js. PATH first, then common install dirs (the server may be
    launched from an IDE whose env lacks the user's PATH)."""
    found = shutil.which("node") or shutil.which("node.exe")
    if found:
        return found
    for c in [r"C:\Program Files\nodejs\node.exe",
              r"C:\Program Files (x86)\nodejs\node.exe",
              "/usr/local/bin/node", "/usr/bin/node",
              os.path.expanduser("~/.nvm/current/bin/node")]:
        if os.path.exists(c):
            return c
    return None


def _node_tool(tool):
    """Locate a Node.js helper like npm or npx.

    Prefer PATH, then common Windows Node install locations and the user's
    global npm bin directory.
    """
    found = shutil.which(tool) or shutil.which(f"{tool}.cmd") or shutil.which(f"{tool}.exe")
    if found:
        return found
    candidates = []
    if os.name == 'nt':
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates += [
                os.path.join(appdata, "npm", f"{tool}.cmd"),
                os.path.join(appdata, "npm", tool),
            ]
        programfiles = os.environ.get("ProgramFiles", r"C:\Program Files")
        programfiles_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        candidates += [
            os.path.join(programfiles, "nodejs", f"{tool}.cmd"),
            os.path.join(programfiles, "nodejs", f"{tool}.exe"),
            os.path.join(programfiles_x86, "nodejs", f"{tool}.cmd"),
            os.path.join(programfiles_x86, "nodejs", f"{tool}.exe"),
        ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


_NODE_BIN = _node_bin()
_LH_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lh_runner.mjs")


def run_lighthouse_node(url, device, throttling, port):
    """Score `url` via the Lighthouse NODE API (lh_runner.mjs), driving the
    SAME authenticated Chrome on `port`. Unlike the CLI, this audits inside the
    logged-in browser context, so login-gated pages load for real instead of
    bouncing to /login. Returns (html, json_str, lhr_dict).

    Raises if Node, the runner, or the lighthouse module can't be found, or if
    the audit produced no report — callers fall back to the CLI / CDP."""
    if not _NODE_BIN:
        raise RuntimeError("Node.js not found on PATH (needed for gated-page Lighthouse).")
    if not os.path.exists(_LH_RUNNER):
        raise RuntimeError(f"Lighthouse runner missing: {_LH_RUNNER}")
    lh_dir = _lighthouse_module_dir()
    if not lh_dir:
        raise RuntimeError("Lighthouse module dir not found (npm i -g lighthouse).")

    tmp = tempfile.mkdtemp(prefix="lhn-")
    out_base = os.path.join(tmp, "report")
    args = [_NODE_BIN, _LH_RUNNER, f"--port={port}", f"--url={url}",
            f"--device={device}", f"--out={out_base}", f"--lhdir={lh_dir}"]
    if not throttling:
        args.append("--no-throttle")
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=200)
        line = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout.strip() else ""
        result = json_lib.loads(line) if line.startswith("{") else {}
        if not result.get("ok"):
            tail = result.get("error") or (proc.stderr or proc.stdout or "")[-800:]
            raise RuntimeError("Node Lighthouse failed: " + str(tail)[-800:])
        with open(result["json"], "r", encoding="utf-8") as fh:
            json_str = fh.read()
        html = ""
        if result.get("html") and os.path.exists(result["html"]):
            with open(result["html"], "r", encoding="utf-8") as fh:
                html = fh.read()
        lhr = json_lib.loads(json_str)
        lhr["runWarnings"] = []
        html = strip_lh_run_warnings(html)
        return html, json_str, lhr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_lighthouse(url, device, throttling, port, cookie_header=None):
    """Run Lighthouse against `url`, attaching to the Chrome already on `port`
    (so logged-in/protected pages score with the live session). Returns
    (html, json_str, lhr_dict)."""
    cli = _lighthouse_cli()
    if not cli:
        raise RuntimeError("Lighthouse CLI not found. Install it: npm i -g lighthouse")

    tmp = tempfile.mkdtemp(prefix="lh-")
    # With dual output (html+json) and --output-path=<base>.json, Lighthouse
    # strips the extension and writes <base>.report.html + <base>.report.json.
    out_base = os.path.join(tmp, "report")
    json_path = out_base + ".report.json"
    html_path = out_base + ".report.html"

    args = [
        cli, url,
        f"--port={port}",
        f"--only-categories={','.join(LH_CATEGORIES)}",
        "--output=html", "--output=json",
        f"--output-path={out_base}",
        "--quiet",
        "--disable-storage-reset",
        "--chrome-flags=--headless",
    ]
    # Lighthouse 12/13 strictly validates that form-factor and screenEmulation
    # agree (else it throws in assertValidSettings). Use the built-in presets:
    #   desktop -> --preset=desktop (sets form-factor + desktop screen together)
    #   mobile  -> the default config IS mobile, so just set the form factor.
    if device == "mobile":
        args.append("--form-factor=mobile")
    else:
        args.append("--preset=desktop")
    # Send the flow's session cookie with Lighthouse's own navigation, so it
    # loads logged-in pages instead of being redirected to the login screen.
    if cookie_header:
        args.append(f'--extra-headers={json_lib.dumps({"Cookie": cookie_header})}')
    if not throttling:
        # "No throttling" = model a fast connection, NOT zero. All-zero throughput
        # makes Lighthouse's estimator divide by zero -> rtt = Infinity, which
        # throws in byte-efficiency audits. Small finite values keep them computing.
        args += [
            "--throttling-method=provided",
            "--throttling.rttMs=1", "--throttling.throughputKbps=100000",
            "--throttling.cpuSlowdownMultiplier=1", "--throttling.requestLatencyMs=1",
            "--throttling.downloadThroughputKbps=100000", "--throttling.uploadThroughputKbps=100000",
        ]

    try:
        # NOTE: do NOT use check=True. On Windows, Lighthouse's bundled
        # chrome-launcher often throws `EPERM` while deleting its own temp dir
        # AFTER the report is already written — the CLI then exits non-zero even
        # though the audit succeeded. So we judge success by whether the JSON
        # report exists, not by the exit code. (This mirrors the original JS
        # server, which wrapped chrome.kill() in try/catch for the same reason.)
        # Point the CLI's temp (where chrome-launcher makes its profile dir) at
        # our per-call folder, so any leftover that triggers the EPERM lives
        # under `tmp` and gets cleaned by the finally-block below.
        env = {**os.environ, "TMP": tmp, "TEMP": tmp}
        proc = subprocess.run(args, capture_output=True, text=True, timeout=180, env=env)
        if not os.path.exists(json_path):
            # Genuine failure — surface Lighthouse's own stderr so it's debuggable.
            tail = (proc.stderr or proc.stdout or "").strip()[-800:]
            raise RuntimeError("Lighthouse produced no report. " + (tail or f"exit {proc.returncode}"))
        html = ""
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as fh:
                html = fh.read()
        with open(json_path, "r", encoding="utf-8") as fh:
            json_str = fh.read()
        lhr = json_lib.loads(json_str)
        # Drop "page was redirected..." runWarnings noise from the lhr AND the
        # HTML's embedded JSON so the banner never renders.
        lhr["runWarnings"] = []
        html = strip_lh_run_warnings(html)
        return html, json_str, lhr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def strip_lh_run_warnings(html):
    """Blank the runWarnings array embedded in the report HTML so the top banner
    never renders. We only touch the embedded report JSON, nothing else."""
    if not html:
        return html
    return re.sub(r'"runWarnings":\s*\[[\s\S]*?\]', '"runWarnings":[]', html, count=1)


# =============================================================================
#  ENGINE: Chrome DevTools (CDP) live capture via Playwright
#  Loads the page in a real browser and reads what it actually experienced:
#  FCP/LCP/CLS (PerformanceObserver), Navigation + Resource Timing, console
#  errors, and a screenshot. Complements Lighthouse's lab numbers.
# =============================================================================
def capture_live_page(page, thresholds, prev=None):
    """Measure whatever page is CURRENTLY loaded (no navigation)."""
    return measure_page(page, thresholds, prev)


def capture_cdp(page, url, thresholds):
    # Clear the network cache + disable it for this navigation so the browser
    # makes REAL requests — that's the only way DNS/Connect/TLS/Wait/Download
    # phases are reported. Without this, a revisited page serves from cache and
    # every request shows 0 ms with no phase breakdown (empty waterfall bars).
    try:
        cdp_session = page.context.new_cdp_session(page)
        cdp_session.send("Network.clearBrowserCache")
        cdp_session.send("Network.setCacheDisabled", {"cacheDisabled": True})
    except Exception:
        cdp_session = None  # non-Chromium or CDP unavailable — proceed anyway
    try:
        page.goto(url, wait_until="load", timeout=60000)
    except Exception:
        pass
    page.wait_for_timeout(1200)  # let late paints / layout shifts register
    result = measure_page(page, thresholds)
    # Re-enable cache so it doesn't bleed into later flow steps unexpectedly.
    try:
        if cdp_session:
            cdp_session.send("Network.setCacheDisabled", {"cacheDisabled": False})
    except Exception:
        pass
    return result


_MEASURE_JS = """(prevState) => {
  const data = { fcp: null, lcp: null, cls: 0, ttfb: null, domContentLoaded: null, load: null, resources: [], transferBytes: 0, totalRequests: null };
  for (const e of performance.getEntriesByType('paint')) {
    if (e.name === 'first-contentful-paint') data.fcp = Math.round(e.startTime);
  }
  const nav = performance.getEntriesByType('navigation')[0];
  if (nav) {
    // Page-level navigation timing phases (for the TIMING breakdown panel):
    // DNS, TCP connect, TLS, request send, server response, DOM processing.
    const np = (a, b) => (a != null && b != null && b >= a) ? Math.round(b - a) : 0;
    data.navPhases = {
      ttfb:     Math.round(nav.responseStart || 0),
      dns:      np(nav.domainLookupStart, nav.domainLookupEnd),
      tcp:      np(nav.connectStart, nav.connectEnd),
      tls:      np(nav.secureConnectionStart, nav.connectEnd),
      request:  np(nav.requestStart, nav.responseStart),
      response: np(nav.responseStart, nav.responseEnd),
      domProc:  np(nav.responseEnd, nav.domContentLoadedEventEnd),
      domDone:  np(nav.domContentLoadedEventEnd, nav.loadEventEnd),
    };
    data.ttfb = Math.round(nav.responseStart);
    data.domContentLoaded = Math.round(nav.domContentLoadedEventEnd);
    data.load = Math.round(nav.loadEventEnd || nav.duration);
  }
  const lcps = performance.getEntriesByType('largest-contentful-paint');
  if (lcps.length) data.lcp = Math.round(lcps[lcps.length - 1].startTime);
  let cls = 0;
  for (const e of performance.getEntriesByType('layout-shift')) {
    if (!e.hadRecentInput) cls += e.value;
  }
  data.cls = Math.round(cls * 1000) / 1000;
  // INP (Interaction to Next Paint): worst interaction latency, from buffered
  // event-timing entries. Null when the user hasn't interacted with this page.
  try {
    let worst = 0;
    for (const e of performance.getEntriesByType('event')) {
      const dur = (e.processingEnd != null && e.startTime != null) ? (e.processingEnd - e.startTime) : (e.duration || 0);
      if (dur > worst) worst = dur;
    }
    data.inp = worst > 0 ? Math.round(worst) : null;
  } catch (_) { data.inp = null; }
  const allRes = performance.getEntriesByType('resource');
  const prevCount = (prevState && prevState.resCount) || 0;
  const stepRes = allRes.slice(prevCount);
  data.stepRequests = stepRes.length;
  data.stepDurationMs = stepRes.length
    ? Math.round(Math.max(...stepRes.map(r => r.responseEnd)) - Math.min(...stepRes.map(r => r.startTime)))
    : 0;
  data.stepTransferBytes = stepRes.reduce((s, r) => s + (r.transferSize || 0), 0);
  data.allResCount = allRes.length;
  const res = allRes.map(r => {
    // Per-request timing phases from PerformanceResourceTiming. Each phase is a
    // duration in ms; some are 0 when reused (keep-alive) or not exposed.
    const rd = (a, b) => (a && b && b >= a) ? Math.round(b - a) : 0;
    const queue = rd(r.startTime, r.fetchStart || r.startTime);
    const dns = rd(r.domainLookupStart, r.domainLookupEnd);
    const tcp = rd(r.connectStart, r.connectEnd);
    const tls = rd(r.secureConnectionStart, r.connectEnd);  // TLS is the tail of connect
    const wait = rd(r.requestStart, r.responseStart);        // server think time (TTFB for this req)
    const download = rd(r.responseStart, r.responseEnd);
    // "Cached" heuristic: a real navigation/transfer reports transferSize; 0 with
    // a non-zero decodedBodySize means it came from cache.
    const cached = (!(r.transferSize > 0) && (r.decodedBodySize > 0));
    return {
      name: r.name, type: r.initiatorType, duration: Math.round(r.duration),
      size: Math.round(r.transferSize || 0), start: Math.round(r.startTime),
      end: Math.round(r.responseEnd || (r.startTime + r.duration)),
      proto: r.nextHopProtocol || '',
      responseStart: Math.round(r.responseStart || 0),
      waitMs: (r.responseStart && r.requestStart) ? Math.round(r.responseStart - r.requestStart) : null,
      phases: { queue, dns, tcp, tls, wait, download, cached },
    };
  });
  // Views of the same requests:
  //  - resources: biggest-first (kept for the existing size table + reports)
  //  - timeline: CHRONOLOGICAL by start time, so a true waterfall lines up on the
  //    time axis (bars offset by start, multi-phase colored). Each item carries
  //    start/end/phases. The slowest is found separately (by duration) for the callout.
  data.resources = res.slice().sort((a, b) => b.size - a.size).slice(0, 30);
  data.timeline = res.slice().sort((a, b) => (a.start || 0) - (b.start || 0)).slice(0, 60);
  data.timelineSpanMs = data.timeline.length
    ? Math.max(...data.timeline.map(r => r.end || 0)) : 0;
  data.transferBytes = res.reduce((sum, r) => sum + r.size, 0);
  data.totalRequests = res.length + 1;
  // INP approximation: the worst interaction latency seen so far (event-timing).
  // Real INP needs ongoing observation; on a measured page this is a fair proxy.
  try {
    const evs = performance.getEntriesByType('event').filter(e => e.duration != null);
    data.inp = evs.length ? Math.round(Math.max(...evs.map(e => e.duration))) : null;
  } catch (e) { data.inp = null; }
  return data;
}"""


def measure_page(page, thresholds, prev=None):
    """Shared measurement: reads PerformanceObserver/timing + a screenshot from
    the page as it currently is, and shapes the CDP result object. `prev`
    (optional) is {resCount,...} from the previous step for per-step SPA deltas."""
    console_errors = []

    def on_console(msg):
        try:
            if msg.type == "error":
                console_errors.append(msg.text)
        except Exception:
            pass

    def on_pageerror(err):
        console_errors.append(str(err))

    page.on("console", on_console)
    page.on("pageerror", on_pageerror)
    page.wait_for_timeout(400)  # settle

    try:
        raw = page.evaluate(_MEASURE_JS, prev or {})
    except Exception:
        raw = {}

    screenshot = None
    try:
        png = page.screenshot(type="jpeg", quality=55)
        screenshot = "data:image/jpeg;base64," + base64.b64encode(png).decode("ascii")
    except Exception:
        pass  # screenshots are best-effort

    try:
        page.remove_listener("console", on_console)
    except Exception:
        pass

    transfer_bytes = raw.get("transferBytes") or 0
    step_transfer = raw.get("stepTransferBytes") or 0
    cdp = {
        "finalUrl": page.url,
        "metrics": {
            "fcp": raw.get("fcp"), "lcp": raw.get("lcp"), "cls": raw.get("cls"),
            "ttfb": raw.get("ttfb"), "domContentLoaded": raw.get("domContentLoaded"), "load": raw.get("load"),
            "inp": raw.get("inp"),
        },
        "navPhases": raw.get("navPhases") or {},
        "totalRequests": raw.get("totalRequests"),
        "transferKb": round(transfer_bytes / 1024) if transfer_bytes else None,
        "stepRequests": raw.get("stepRequests"),
        "stepDurationMs": raw.get("stepDurationMs"),
        "stepTransferKb": round(step_transfer / 1024) if step_transfer else 0,
        "_resBaseline": raw.get("allResCount") or 0,
        "consoleErrors": len(console_errors),
        "consoleErrorSamples": console_errors[:5],
        "resources": raw.get("resources") or [],
        "timeline": raw.get("timeline") or [],
        "timelineSpanMs": raw.get("timelineSpanMs") or 0,
        "screenshot": screenshot,
    }
    m = cdp["metrics"]

    # --- Perceived ("true") response time, the number a single user actually feels ---
    # A server can reply in (say) 5s — that's TTFB — but the page isn't usable until the
    # browser paints the main content (LCP) and finishes loading. So perceived time is the
    # later of LCP / load, and render time is the gap the browser spent on top of the server.
    ttfb = m.get("ttfb") or 0
    lcp = m.get("lcp") or 0
    load = m.get("load") or 0
    perceived = max(lcp, load, ttfb)
    render = max(perceived - ttfb, 0)
    cdp["responseBreakdown"] = {
        "serverMs": ttfb or None,        # time to first byte — the server's part
        "renderMs": render or None,      # browser render/scripting on top
        "perceivedMs": perceived or None,  # what the user waits for, end to end
        "basis": "lcp" if lcp >= load else "load",  # which milestone perceived is based on
    }
    # The single request that took longest. Only requests with a REAL measured
    # duration count — a 0 ms entry means timing wasn't captured (served from
    # cache or a cross-origin resource without Timing-Allow-Origin), so it must
    # NOT be reported as "slowest". If nothing has timing, say so honestly.
    tl = cdp["timeline"]
    timed = [r for r in tl if (r.get("duration") or 0) > 0]
    if timed:
        slow = max(timed, key=lambda r: r.get("duration") or 0)
        cdp["slowestRequest"] = {
            "name": slow.get("name"), "type": slow.get("type"),
            "durationMs": slow.get("duration"), "sizeBytes": slow.get("size"),
            "startMs": slow.get("start"),
        }
    else:
        cdp["slowestRequest"] = None
        cdp["timingUnavailable"] = bool(tl)  # we had requests, but none reported timing
    cdp["bands"] = {
        "fcp": value_band("fcp", m["fcp"], thresholds),
        "lcp": value_band("lcp", m["lcp"], thresholds),
        "cls": value_band("cls", m["cls"], thresholds),
        "ttfb": value_band("ttfb", m["ttfb"], thresholds),
        "load": value_band("pageLoad", m["load"], thresholds),
        "requests": value_band("requests", cdp["totalRequests"], thresholds),
        "transferKb": value_band("transferKb", cdp["transferKb"], thresholds),
        "consoleErrors": value_band("consoleErrors", cdp["consoleErrors"], thresholds),
    }
    return cdp


# Launch a system browser. `prefer` is one of auto|chrome|msedge|chromium.
# Playwright's own chromium download is often blocked on corporate networks, so
# `auto` falls back through Chrome -> Edge -> bundled chromium.
BROWSER_CHANNELS = {
    "auto":     [{"channel": "chrome", "name": "Google Chrome"}, {"channel": "msedge", "name": "Microsoft Edge"}, {"name": "Chromium (bundled)"}],
    "chrome":   [{"channel": "chrome", "name": "Google Chrome"}],
    "msedge":   [{"channel": "msedge", "name": "Microsoft Edge"}],
    "chromium": [{"name": "Chromium (bundled)"}],
}
LAST_BROWSER = None  # records which browser the most recent launch used (UI)


def launch_browser(pw, extra_args=None, prefer="auto", headed=False):
    """Launch a system browser. `headed=True` opens a VISIBLE window."""
    global LAST_BROWSER
    extra_args = extra_args or []
    last_err = None
    order = BROWSER_CHANNELS.get(prefer) or BROWSER_CHANNELS["auto"]
    for opt in order:
        try:
            kwargs = {"headless": not headed, "args": extra_args}
            if opt.get("channel"):
                kwargs["channel"] = opt["channel"]
            b = pw.chromium.launch(**kwargs)
            LAST_BROWSER = opt["name"] + (" (visible)" if headed else "")
            return b
        except Exception as err:
            last_err = err
    raise last_err or RuntimeError(
        "No usable browser. Install Chrome/Edge, or run: python -m playwright install chromium"
    )


# =============================================================================
#  RESULT SHAPING + PERSISTENCE
# =============================================================================
def _safe_name(name, limit=50):
    return re.sub(r"[^a-z0-9]", "_", str(name), flags=re.I)[:limit]


def save_report(name, html):
    stamp = now_iso().replace(":", "-").replace(".", "-")
    filename = f"{_safe_name(name)}-{stamp}.html"
    with open(os.path.join(REPORTS_DIR, filename), "w", encoding="utf-8") as fh:
        fh.write(html)
    return f"/reports/{filename}"


def save_file(base_name, ext, content):
    stamp = now_iso().replace(":", "-").replace(".", "-")
    filename = f"{_safe_name(base_name)}-{stamp}.{ext}"
    with open(os.path.join(REPORTS_DIR, filename), "w", encoding="utf-8") as fh:
        fh.write(content)
    return f"/reports/{filename}"


def save_pdf(pw, base_name, html):
    """Render an HTML string to a PDF using the headless browser (best-effort)."""
    browser = None
    try:
        browser = launch_browser(pw)
        page = browser.new_context().new_page()
        page.set_content(html, wait_until="load")
        stamp = now_iso().replace(":", "-").replace(".", "-")
        filename = f"{_safe_name(base_name)}-{stamp}.pdf"
        page.pdf(
            path=os.path.join(REPORTS_DIR, filename), format="A4", print_background=True,
            margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"},
        )
        return f"/reports/{filename}"
    except Exception:
        return None  # PDF is best-effort; never fail the run over it.
    finally:
        if browser:
            try:
                browser.close()
            except Exception:
                pass


def esc(s):
    return html_lib.escape("" if s is None else str(s), quote=True)


def score_color(s):
    if s is None:
        return "#9aa0a6"
    return "#0c8a3e" if s >= 90 else "#b8860b" if s >= 50 else "#c5221f"


def score_ring(score, size=92):
    """A coloured donut/ring SVG for a 0-100 score."""
    c = score_color(score)
    r = (size - 12) / 2
    circ = 2 * math.pi * r
    pct = 0 if score is None else score / 100
    dash = f"{circ * pct:.1f}"
    label = "—" if score is None else score
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="#eef0f3" stroke-width="9"/>'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{c}" stroke-width="9" stroke-linecap="round" '
        f'stroke-dasharray="{dash} {circ}" transform="rotate(-90 {size/2} {size/2})"/>'
        f'<text x="50%" y="50%" text-anchor="middle" dy=".34em" font-size="{size*0.3}" font-weight="800" fill="{c}">{label}</text>'
        f"</svg>"
    )


def _metric_val(page, key):
    for m in page.get("metrics") or []:
        if m["key"] == key:
            return m.get("value")
    return "—"


def build_report_narrative(flow_name, pages):
    """A plain-English overview for a non-technical reader. LLM-written when a key
    is configured; otherwise a deterministic summary from the real data. Returns
    (text, engine_label). Cached on the pages list so HTML+MD reuse one LLM call."""
    # Reuse a narrative already computed for this same run (HTML then MD).
    if pages and isinstance(pages[0], dict) and pages[0].get("_narrative"):
        cached = pages[0]["_narrative"]
        return cached["text"], cached["engine"]
    ok = [p for p in pages if not p.get("error")]

    def perceived(p):
        return ((p.get("cdp") or {}).get("responseBreakdown") or {}).get("perceivedMs")

    worst = max(ok, key=lambda p: perceived(p) or 0, default=None)
    # Deterministic fallback first (always works, no LLM).
    parts = [f"This report covers {len(pages)} page(s) in “{flow_name}”."]
    if worst and perceived(worst):
        rb = (worst.get("cdp") or {}).get("responseBreakdown") or {}
        s, r, t = rb.get("serverMs"), rb.get("renderMs"), rb.get("perceivedMs")
        sec = lambda ms: f"{ms/1000:.1f}s" if ms and ms >= 1000 else (f"{ms} ms" if ms else "—")
        parts.append(
            f"The slowest page to feel usable was “{worst.get('name')}”, where the server "
            f"replied in {sec(s)} but the browser took {sec(r)} more to render — so a single "
            f"user waited about {sec(t)} in total. The wait is mostly {'rendering in the browser' if (r or 0) > (s or 0) else 'the server response'}."
        )
        slow = (worst.get("cdp") or {}).get("slowestRequest") or {}
        if slow.get("name"):
            parts.append(f"The single slowest request was {slow.get('type','a request')} taking {slow.get('durationMs')} ms.")
    deterministic = " ".join(parts)

    def _cache(text, engine):
        if pages and isinstance(pages[0], dict):
            pages[0]["_narrative"] = {"text": text, "engine": engine}
        return text, engine

    if not ai_active():
        return _cache(deterministic, "rule-based")
    # LLM narrative over the whole run.
    digest = {"flow": flow_name, "pages": [
        {"name": p.get("name"), "score": p.get("performanceScore"),
         "perceivedMs": perceived(p),
         "serverMs": ((p.get("cdp") or {}).get("responseBreakdown") or {}).get("serverMs"),
         "renderMs": ((p.get("cdp") or {}).get("responseBreakdown") or {}).get("renderMs"),
         "slowestRequest": (p.get("cdp") or {}).get("slowestRequest")}
        for p in ok]}
    system = (
        "You are a web-performance analyst. Write a short (3-5 sentence) plain-English summary "
        "for a NON-TECHNICAL reader of this single-user performance run. Explain how fast it felt, "
        "that perceived wait = server response + browser rendering, name the slowest page and the "
        "slowest request, and end with the single most useful fix. Plain prose only, no markdown."
    )
    try:
        txt = llm_chat(system, json_lib.dumps(digest, indent=2), max_tokens=500, temperature=0.3)
        return _cache(txt.strip(), "azure-openai")
    except Exception:
        return _cache(deterministic, "rule-based")


def build_timing_resources_panels(cdp):
    """Two side-by-side panels for a per-page card:
      ⏱ TIMING    — navigation phases (TTFB/DNS/TCP/SSL/Request/Response/DOM) as bars
      📦 RESOURCES — request count + bytes grouped by type, as bars
    Both are plain, labelled, and self-explanatory for a non-technical reader."""
    nav = cdp.get("navPhases") or {}
    # Timing rows: (label, ms, colour). Order mirrors how a load actually unfolds.
    timing = [
        ("TTFB", nav.get("ttfb"), "#3b82f6"),
        ("DNS", nav.get("dns"), "#0aa6b8"),
        ("TCP", nav.get("tcp"), "#f0a000"),
        ("SSL/TLS", nav.get("tls"), "#9b59b6"),
        ("Request", nav.get("request"), "#e0457b"),
        ("Response", nav.get("response"), "#e86b1c"),
        ("DOM Proc", nav.get("domProc"), "#1aa260"),
        ("DOM Done", nav.get("domDone"), "#7c5cff"),
    ]
    timing = [(l, v, c) for (l, v, c) in timing if v is not None]
    timing_html = ""
    if timing:
        mx = max([v for _, v, _ in timing] + [1])
        rows = "".join(
            f'<div class="tm-row"><span class="tm-l">{esc(l)}</span>'
            f'<span class="tm-bar"><span style="width:{max(2,(v/mx)*100):.1f}%;background:{c}"></span></span>'
            f'<span class="tm-v">{v} ms</span></div>'
            for l, v, c in timing
        )
        timing_html = (f'<div class="rp-card"><div class="rp-h">⏱️ TIMING</div>'
                       f'<div class="rp-sub">Where the load time went — from finding the server (DNS) '
                       f'to the server replying (TTFB) to the browser building the page (DOM).</div>{rows}</div>')

    # Resources by type: count + total bytes per initiatorType.
    by_type = {}
    for r in cdp.get("resources") or []:
        t = (r.get("type") or "other").lower()
        e = by_type.setdefault(t, {"n": 0, "bytes": 0})
        e["n"] += 1
        e["bytes"] += r.get("size") or 0
    res_html = ""
    if by_type:
        order = sorted(by_type.items(), key=lambda kv: kv[1]["n"], reverse=True)
        mxn = max([v["n"] for _, v in order] + [1])
        rows = "".join(
            f'<div class="tm-row"><span class="tm-l">{esc(t)}</span>'
            f'<span class="tm-bar"><span style="width:{max(2,(v["n"]/mxn)*100):.1f}%;background:#5b2be0"></span></span>'
            f'<span class="tm-v">{v["n"]}{(" · " + str(round(v["bytes"]/1024)) + " KB") if v["bytes"] else ""}</span></div>'
            for t, v in order
        )
        res_html = (f'<div class="rp-card"><div class="rp-h">📦 RESOURCES</div>'
                    f'<div class="rp-sub">What the page downloaded, grouped by kind (images, scripts, styles). '
                    f'More/heavier files mean a slower page.</div>{rows}</div>')

    if not timing_html and not res_html:
        return ""
    return f'<div class="rp-wrap">{timing_html}{res_html}</div>'


def build_replay_carousel(pages):
    """A 'Traversal Replay' carousel: step through each page's screenshot with
    Prev / Play / Next, so a reviewer can SEE the journey the single user took.
    Self-contained (inline JS), works in the saved HTML report."""
    shots = [(i, p.get("name"), (p.get("cdp") or {}).get("screenshot"))
             for i, p in enumerate(pages) if (p.get("cdp") or {}).get("screenshot")]
    if not shots:
        return ""
    slides = "".join(
        f'<img class="rep-img" data-idx="{n}" src="{src}" style="display:{"block" if n == 0 else "none"}" alt="{esc(name or "")}">'
        for n, (i, name, src) in enumerate(shots)
    )
    total = len(shots)
    cap0 = esc(shots[0][1] or "Page 1")
    # Inline script: Prev/Play/Next over the slides. IIFE so multiple reports don't clash.
    script = (
        "<script>(function(){var W=document.currentScript.closest('.rep');var imgs=W.querySelectorAll('.rep-img');"
        "var cap=W.querySelector('.rep-cap');var pos=W.querySelector('.rep-pos');var t=null;var i=0;var n=imgs.length;"
        "var names=[" + ",".join(f'"{esc(s[1] or ("Page "+str(k+1)))}"' for k, s in enumerate(shots)) + "];"
        "function show(x){i=(x+n)%n;imgs.forEach(function(im,k){im.style.display=k===i?'block':'none'});"
        "cap.textContent=names[i];pos.textContent=(i+1)+' / '+n;}"
        "W.querySelector('.rep-prev').onclick=function(){clearInterval(t);t=null;show(i-1);};"
        "W.querySelector('.rep-next').onclick=function(){clearInterval(t);t=null;show(i+1);};"
        "W.querySelector('.rep-play').onclick=function(e){if(t){clearInterval(t);t=null;e.target.textContent='\\u25B6 Play';}"
        "else{e.target.textContent='\\u23F8 Pause';t=setInterval(function(){show(i+1);},1500);}};"
        "})();</script>"
    )
    return (
        '<div class="panel rep">'
        f'<h2>🎬 Traversal replay <span class="muted" style="font-weight:400;font-size:13px">— step through the journey ({total} pages)</span></h2>'
        '<div class="rp-sub" style="margin:-6px 0 12px">This is exactly what the single user saw on each step, in order. '
        'Press Play to watch the journey, or use Prev / Next.</div>'
        f'<div class="rep-stage">{slides}</div>'
        '<div class="rep-ctl">'
        '<button class="rep-prev">« Prev</button>'
        '<button class="rep-play">▶ Play</button>'
        '<button class="rep-next">Next »</button>'
        f'<span class="rep-pos">1 / {total}</span>'
        f'<span class="rep-cap">{cap0}</span>'
        '</div>' + script + '</div>'
    )


def build_page_ai_block(p, i):
    """Per-page 'AI insights' panel. Always rendered, but the LLM summary is
    generated ON DEMAND: each page shows a '✨ Get AI insights' button that calls
    the running server (/api/page-insights) and renders the result inline. No LLM
    is called at report-build time, so reports stay data-only until a reader asks.

    The page's audit digest is embedded as JSON in a data-attribute so the button
    has everything it needs to ask for insights without re-running the audit."""
    name = esc(p.get("name") or "this page")
    # Same compact digest the LLM analysis uses — built for every page regardless
    # of any opt-in, since insights are now requested per-page after the fact.
    digest = _analysis_digest(
        p.get("name"), p.get("finalUrl") or p.get("url"),
        p.get("performanceScore"), p.get("verdict"),
        p.get("metrics"), p.get("opportunities"), p.get("cdp"),
    )
    digest_json = esc(json_lib.dumps(digest))
    return (
        f'<div class="ai-block" data-ai-idx="{i}" data-ai-digest="{digest_json}">'
        '<div class="ai-row">'
        '<span class="ai-badge">✨ AI insights</span>'
        f'<span class="ai-sum">🤖 Add AI insights — write a plain-English summary of this page&apos;s vitals, timing, and waterfall. Leave off for a data-only report.</span>'
        f'<button type="button" class="ai-get" onclick="getPageInsights(this)">✨ Get AI insights</button>'
        '</div>'
        '<div class="ai-out" style="display:none"></div>'
        '</div>'
    )


def build_journey_report(flow_name, pages, device):
    """Build ONE self-contained, colourful dashboard report with EVERYTHING
    inline: hero score gauge, KPI cards, a summary table, and per-transaction
    cards with Lighthouse metric chips + CDP live tiles + a network waterfall."""
    # No AI on the front (hero) section — AI insights live per-page, on demand
    # (see build_page_ai_block). The report opens data-only.
    ok = [p for p in pages if not p.get("error") and p.get("performanceScore") is not None]
    avg_score = round(sum(p["performanceScore"] for p in ok) / len(ok)) if ok else None
    if avg_score is None:
        v = "No data"
    elif avg_score >= 90:
        v = "Excellent — fast & healthy"
    elif avg_score >= 50:
        v = "Needs improvement"
    else:
        v = "Poor — significant issues"
    total_req = sum((p.get("cdp") or {}).get("totalRequests") or 0 for p in ok)
    total_kb = sum((p.get("cdp") or {}).get("transferKb") or 0 for p in ok)
    total_err = sum((p.get("cdp") or {}).get("consoleErrors") or 0 for p in ok)

    def num(p, k):
        for x in p.get("metrics") or []:
            if x["key"] == k:
                return x.get("numeric")
        return None

    avg_lcp = round(sum(num(p, "lcp") or 0 for p in ok) / len(ok)) if ok else None

    def kpi(label, value, sub=None, color=None):
        sub_html = f'<div class="kpi-s">{sub}</div>' if sub else ""
        return (f'<div class="kpi"><div class="kpi-v" style="color:{color or "#1a1a2e"}">{value}</div>'
                f'<div class="kpi-l">{label}</div>{sub_html}</div>')

    # Summary table rows.  Columns (matching a single-user perf report):
    #   # | Transaction | Score | LCP | FCP | CLS | TTFB | DCL | Load | Interact | Req | Transfer
    def _ms(v):
        return f"{v} ms" if v is not None else "—"

    sum_rows = []
    for i, p in enumerate(pages):
        if p.get("error"):
            sum_rows.append(f'<tr><td>{i+1}</td><td>{esc(p["name"])}</td>'
                            f'<td colspan="10" style="color:#c5221f">⚠ {esc(p["error"])}</td></tr>')
            continue
        c = p.get("cdp") or {}
        cm = c.get("metrics") or {}
        if p.get("lighthouseSkipped"):
            score_cell = '<span class="badge live">CDP ✓</span>'
            lcp_v, fcp_v = _ms(cm.get("lcp")), _ms(cm.get("fcp"))
            cls_v = cm.get("cls") if cm.get("cls") is not None else "—"
        else:
            ps = p.get("performanceScore")
            score_cell = f'<span class="badge" style="background:{score_color(ps)}">{ps if ps is not None else "—"}</span>'
            lcp_v, fcp_v, cls_v = _metric_val(p, "lcp"), _metric_val(p, "fcp"), _metric_val(p, "cls")
        inp_v = cm.get("inp")
        sum_rows.append(
            f'<tr><td class="muted">{i+1}</td>'
            f'<td><a href="#page-{i}"><b>{esc(p["name"])}</b></a></td>'
            f'<td>{score_cell}</td>'
            f'<td>{lcp_v}</td><td>{fcp_v}</td><td>{cls_v}</td>'
            f'<td>{_ms(cm.get("ttfb"))}</td>'
            f'<td>{_ms(cm.get("domContentLoaded"))}</td>'
            f'<td>{_ms(cm.get("load"))}</td>'
            f'<td>{_ms(inp_v) if inp_v is not None else "—"}</td>'
            f'<td>{c.get("totalRequests") if c.get("totalRequests") is not None else "—"}</td>'
            f'<td>{(str(c.get("transferKb")) + " KB") if c.get("transferKb") is not None else "—"}</td></tr>'
        )
    sum_rows_html = "\n".join(sum_rows)

    # Per-transaction detail cards.
    detail_cards = []
    for i, p in enumerate(pages):
        if p.get("error"):
            detail_cards.append(
                f'<section id="page-{i}" class="tcard"><div class="tcard-head" '
                f'style="background:linear-gradient(135deg,#c5221f,#7a1512)">'
                f'<div class="th-name">{i+1}. {esc(p["name"])}</div></div>'
                f'<div class="tcard-body"><p style="color:#c5221f">⚠ {esc(p["error"])}</p></div></section>'
            )
            continue

        chips = "".join(
            f'<div class="chip" style="border-color:{score_color(m.get("score"))}">'
            f'<span class="chip-k">{esc(m["label"])}</span><span class="chip-v">{esc(m["value"])}</span></div>'
            for m in (p.get("metrics") or [])
        )
        cdp = p.get("cdp")
        cdp_block = '<p class="muted">No CDP data captured.</p>'
        if cdp and cdp.get("metrics"):
            cm = cdp["metrics"]
            bands = cdp.get("bands") or {}

            def ms(v):
                return "n/a" if v is None else f"{v} ms"

            # Full vitals tile grid (matches the reference report): the live numbers
            # a single user experienced, each colour-banded Good / Needs work / Poor.
            cdp_block = (
                '<div class="tiles">'
                + band_tile("FCP", ms(cm.get("fcp")), bands.get("fcp"))
                + band_tile("LCP", ms(cm.get("lcp")), bands.get("lcp"))
                + band_tile("CLS", cm.get("cls"), bands.get("cls"))
                + band_tile("TTFB", ms(cm.get("ttfb")), bands.get("ttfb"))
                + band_tile("DOM Int.", ms(cm.get("domContentLoaded")), None)
                + band_tile("DOM Done", ms(cm.get("load")), bands.get("load"))
                + band_tile("Load", ms(cm.get("load")), bands.get("load"))
                + band_tile("Interact (INP)", (ms(cm.get("inp")) if cm.get("inp") is not None else "—"), None)
                + band_tile("Requests", cdp.get("totalRequests"), bands.get("requests"))
                + band_tile("Transferred", (f'{cdp["transferKb"]} KB' if cdp.get("transferKb") is not None else "—"), bands.get("transferKb"))
                + (band_tile("Console errors", cdp.get("consoleErrors"), bands.get("consoleErrors")) if (cdp.get("consoleErrors") or 0) > 0 else "")
                + "</div>"
            )
            # ⏱ TIMING breakdown + 📦 RESOURCES-by-type (side by side), like a single-user trace.
            cdp_block += build_timing_resources_panels(cdp)
            # Perceived ("true") response time: server (TTFB) + browser render = what a user waits for.
            rb = cdp.get("responseBreakdown") or {}
            if rb.get("perceivedMs"):
                def _sec(v):
                    return "—" if v is None else (f"{v/1000:.1f} s" if v >= 1000 else f"{v} ms")
                note = (f'The server replied in <b>{_sec(rb.get("serverMs"))}</b>, then the browser took '
                        f'<b>{_sec(rb.get("renderMs"))}</b> to render — a real visitor waited about '
                        f'<b>{_sec(rb.get("perceivedMs"))}</b>.') if rb.get("serverMs") is not None else \
                       (f'A real visitor waited about <b>{_sec(rb.get("perceivedMs"))}</b>.')
                cdp_block += (
                    '<div class="wf-h">⏱️ Perceived response time — what one user actually waits for</div>'
                    '<div class="tiles">'
                    + band_tile("Server (TTFB)", _sec(rb.get("serverMs")), None)
                    + band_tile("Rendering", _sec(rb.get("renderMs")), None)
                    + band_tile("Total perceived", _sec(rb.get("perceivedMs")), None)
                    + '</div>'
                    + f'<p class="muted" style="margin:8px 0 0">{note}</p>'
                )
            if (cdp.get("consoleErrors") or 0) > 0 and cdp.get("consoleErrorSamples"):
                rows = "".join(
                    f'<div class="cerr-row">{esc(str(s)[:240])}</div>'
                    for s in cdp["consoleErrorSamples"]
                )
                cdp_block += (f'<div class="cerr"><div class="cerr-h">⚠️ Console errors on this page '
                              f'({cdp["consoleErrors"]})</div>{rows}</div>')
            # DevTools-style waterfall: requests in chronological order, each bar
            # placed on a shared time axis and split into timing phases
            # (DNS/Connect/TLS/Wait/Download). The URL is shown on every row.
            tl = (cdp.get("timeline") or [])[:20]
            if tl:
                slow = cdp.get("slowestRequest") or {}
                span = max(cdp.get("timelineSpanMs") or 0, 1)
                phase_defs = [("queue", "#c7ccd6"), ("dns", "#0aa6b8"), ("tcp", "#f0a000"),
                              ("tls", "#9b59b6"), ("wait", "#e0457b"), ("download", "#1aa260")]
                wf_rows = ""
                for r in tl:
                    full = r.get("name", "")
                    short = esc((full.split("/")[-1] or full)[:54]) or "(document)"
                    left = ((r.get("start") or 0) / span) * 100
                    ph = r.get("phases") or {}
                    segs = "".join(
                        f'<span style="width:{(ph.get(k, 0) / span) * 100:.2f}%;background:{col}"></span>'
                        for k, col in phase_defs if (ph.get(k) or 0) > 0
                    )
                    if not segs:
                        col = "#b7bdc8" if ph.get("cached") else "#5b2be0"
                        segs = f'<span style="width:{max(0.4, ((r.get("duration") or 0) / span) * 100):.2f}%;background:{col}"></span>'
                    is_slow = slow.get("name") and r.get("name") == slow.get("name")
                    row_cls = "wf-row2 slow" if is_slow else "wf-row2"
                    size_txt = (f'{round(r["size"]/1024)} KB' if r.get("size") else ("cached" if ph.get("cached") else "—"))
                    wf_rows += (
                        f'<div class="{row_cls}"><span class="wf-c-type">{esc(r.get("type") or "other")}</span>'
                        f'<span class="wf-c-url" title="{esc(full)}">{short}</span>'
                        f'<span class="wf-c-proto">{esc(r.get("proto") or "—")}</span>'
                        f'<span class="wf-c-size">{size_txt}</span>'
                        f'<span class="wf-c-dur">{r.get("duration") or 0} ms</span>'
                        f'<span class="wf-c-track"><span class="wf-seg" style="margin-left:{left:.2f}%">{segs}</span></span></div>'
                    )
                ticks = [round(span * i / 6) for i in range(7)]
                axis = ('<div class="wf-axis"><span></span><span></span><span></span><span></span><span></span>'
                        '<span class="wf-axis-r">' + "".join(f'<i>{t} ms</i>' for t in ticks) + '</span></div>')
                legend = ('<div class="wf-legend">'
                          '<span><i style="background:#0aa6b8"></i>DNS</span>'
                          '<span><i style="background:#f0a000"></i>Connect</span>'
                          '<span><i style="background:#9b59b6"></i>TLS</span>'
                          '<span><i style="background:#e0457b"></i>Wait (TTFB)</span>'
                          '<span><i style="background:#1aa260"></i>Download</span>'
                          '<span><i style="background:#b7bdc8"></i>Cached</span></div>')
                head = ('<div class="wf-row2 wf-h2"><span class="wf-c-type">TYPE</span><span class="wf-c-url">RESOURCE</span>'
                        '<span class="wf-c-proto">PROTO</span><span class="wf-c-size">SIZE</span>'
                        '<span class="wf-c-dur">DURATION</span><span class="wf-c-track">TIMELINE</span></div>')
                cdp_block += (f'<div class="wf-h">🌊 Request waterfall — {span} ms timeline</div>'
                              f'<div class="rp-sub" style="margin:0 0 8px">Every file the page requested, in the order it '
                              f'happened. Each bar shows that file\'s journey: finding the server, connecting, waiting, '
                              f'then downloading. The longest bar is what held the page up most.</div>'
                              f'<div class="wf2">{head}{axis}{wf_rows}</div>{legend}')
                if slow.get("name"):
                    cdp_block += (f'<p class="muted" style="margin:6px 0 0">🐢 Slowest request: '
                                  f'<b>{esc(slow.get("name"))[:80]}</b> — {slow.get("durationMs")} ms '
                                  f'({esc(slow.get("type") or "doc")}).</p>')
                elif cdp.get("timingUnavailable"):
                    cdp_block += ('<p class="muted" style="margin:6px 0 0">ℹ️ Per-request timing was not '
                                  'available (cache / cross-origin).</p>')

        head_color = "#5b2be0" if p.get("lighthouseSkipped") else score_color(p.get("performanceScore"))
        head_badge = ('<div class="cdp-pill">CDP&nbsp;✓<br><span>live</span></div>'
                      if p.get("lighthouseSkipped") else score_ring(p.get("performanceScore"), 70))
        shot_block = ""
        if cdp and cdp.get("screenshot"):
            shot_block = (f'<div class="sec-h">📸 Captured page</div>'
                          f'<img class="page-shot" src="{cdp["screenshot"]}" alt="{esc(p["name"])}">')
        if p.get("lighthouseSkipped"):
            lh_section = ('<div class="sec-h">🔦 Lighthouse</div>'
                          '<div class="ops ok-ops" style="background:#eef0ff;color:#5b2be0">'
                          'ℹ️ Login-gated page — measured live via Chrome DevTools below '
                          "(Lighthouse can't cold-load a logged-in page).</div>")
        else:
            lh_section = f'<div class="sec-h">🔦 Lighthouse metrics</div><div class="chips">{chips}</div>'
        report_link = (f'<p style="margin-top:12px"><a href="{p["reportUrl"]}" target="_blank">'
                       f"Open full Lighthouse report ↗</a></p>") if p.get("reportUrl") else ""
        detail_cards.append(
            f'<section id="page-{i}" class="tcard">'
            f'<div class="tcard-head" style="background:linear-gradient(135deg,{head_color},{head_color}cc)">'
            f"<div>{head_badge}</div>"
            f'<div class="th-info"><div class="th-name">{i+1}. {esc(p["name"])}</div>'
            f'<div class="th-url">{esc(p.get("finalUrl") or p.get("url"))}</div></div></div>'
            f'<div class="tcard-body">{build_page_ai_block(p, i)}{shot_block}{lh_section}'
            f'<div class="sec-h">🛠️ Chrome DevTools (live)</div>{cdp_block}{report_link}'
            f"</div></section>"
        )
    detail_html = "\n".join(detail_cards)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Journey report — {esc(flow_name)}</title>
  <style>
    *{{box-sizing:border-box}}
    body{{font:14px/1.55 'Segoe UI',-apple-system,Roboto,sans-serif;margin:0;background:#eef1f6;color:#1a1a2e}}
    .wrap{{max-width:1080px;margin:0 auto;padding:28px 20px 56px}}
    a{{color:#5b5bff;text-decoration:none}} a:hover{{text-decoration:underline}}
    .muted{{color:#8a90a2}}
    .hero{{background:linear-gradient(135deg,#5b2be0 0%,#8b2fd6 50%,#c5267f 100%);color:#fff;border-radius:20px;padding:30px 34px;
      display:flex;align-items:center;gap:28px;box-shadow:0 16px 40px rgba(91,43,224,.32);margin-bottom:18px;flex-wrap:wrap}}
    .hero-gauge svg circle:first-child{{stroke:rgba(255,255,255,.25)}}
    .hero h1{{margin:0 0 6px;font-size:26px;font-weight:800;letter-spacing:-.3px}}
    .hero .meta{{opacity:.9;font-size:13.5px}}
    .hero .verdict{{display:inline-block;margin-top:10px;background:rgba(255,255,255,.18);padding:5px 14px;border-radius:20px;font-weight:600;font-size:13px;backdrop-filter:blur(4px)}}
    .kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:22px}}
    .kpi{{background:#fff;border-radius:16px;padding:18px;text-align:center;box-shadow:0 4px 14px rgba(20,20,60,.06)}}
    .kpi-v{{font-size:26px;font-weight:800;line-height:1}} .kpi-l{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#8a90a2;margin-top:6px}} .kpi-s{{font-size:11px;color:#b0b4c2;margin-top:2px}}
    .panel{{background:#fff;border-radius:18px;padding:22px 26px;margin-bottom:20px;box-shadow:0 4px 14px rgba(20,20,60,.06)}}
    .panel h2{{margin:0 0 14px;font-size:17px}}
    table{{width:100%;border-collapse:collapse}} th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid #f0f1f5}}
    th{{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#8a90a2}} tr:hover td{{background:#fafbff}}
    .badge{{display:inline-block;min-width:34px;text-align:center;padding:3px 11px;border-radius:20px;color:#fff;font-weight:700;font-size:13px}}
    .badge.live{{background:#5b2be0}}
    .page-shot{{width:100%;border-radius:12px;border:1px solid #e7e9f0;margin:4px 0 8px;box-shadow:0 2px 10px rgba(20,20,60,.08)}}
    .cerr{{margin-top:12px;background:#fff5f5;border:1px solid #ffd7d7;border-radius:10px;padding:12px 14px}}
    .cerr-h{{font-weight:700;font-size:13px;color:#c5221f;margin-bottom:6px}}
    .cerr-row{{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#7a1512;padding:3px 0;border-bottom:1px dashed #ffe0e0;word-break:break-all}}
    .cerr-row:last-child{{border:0}}
    .cdp-pill{{width:70px;height:70px;border-radius:50%;background:rgba(255,255,255,.18);display:flex;flex-direction:column;align-items:center;justify-content:center;color:#fff;font-weight:800;font-size:15px;line-height:1.1}}
    .cdp-pill span{{font-size:11px;font-weight:600;opacity:.9}}
    .tcard{{background:#fff;border-radius:18px;overflow:hidden;margin-bottom:20px;box-shadow:0 6px 20px rgba(20,20,60,.08)}}
    .tcard-head{{display:flex;align-items:center;gap:18px;padding:18px 24px;color:#fff}}
    .tcard-head svg text{{fill:#fff}} .tcard-head svg circle:first-child{{stroke:rgba(255,255,255,.3)}} .tcard-head svg circle:last-child{{stroke:#fff}}
    .th-name{{font-size:18px;font-weight:700}} .th-url{{font-size:12.5px;opacity:.9;word-break:break-all}}
    .tcard-body{{padding:20px 24px}}
    .sec-h{{font-size:13px;font-weight:700;color:#5b2be0;text-transform:uppercase;letter-spacing:.04em;margin:18px 0 10px}}
    .sec-h:first-child{{margin-top:0}}
    .chips{{display:flex;flex-wrap:wrap;gap:10px}}
    .chip{{border:2px solid #ccc;border-radius:12px;padding:8px 14px;min-width:92px}} .chip-k{{display:block;font-size:11px;color:#8a90a2;text-transform:uppercase}} .chip-v{{display:block;font-size:17px;font-weight:700}}
    .ops{{margin:14px 0;background:#fff8ec;border-radius:12px;padding:12px 16px}} .ops-h{{font-weight:700;font-size:13px;margin-bottom:6px;color:#b8860b}}
    .op{{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px dashed #f0e6cf;font-size:13px}} .op:last-child{{border:0}} .op-s{{color:#b8860b;font-weight:600;white-space:nowrap;margin-left:10px}}
    .ok-ops{{background:#eaf7ee;color:#0c8a3e;font-weight:600}}
    .tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
    .tile{{background:#f8f9fd;border-radius:12px;padding:12px 14px;border-left:5px solid var(--bc)}} .tl{{font-size:11px;color:#8a90a2;text-transform:uppercase}} .tv{{font-size:21px;font-weight:800;margin:3px 0}} .tr{{font-size:11px;font-weight:600}}
    .wf-h{{font-size:13px;font-weight:700;margin:18px 0 10px}}
    .wf-row{{display:flex;align-items:center;gap:10px;margin-bottom:6px;font-size:12px}}
    .wf-type{{width:54px;color:#8a90a2;text-transform:uppercase;font-size:10.5px;flex:none}}
    .wf-url{{width:240px;flex:none;color:#3c4043;font-family:ui-monospace,Consolas,monospace;font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
    .wf-bar{{flex:1;background:#eef1f6;border-radius:6px;height:14px;position:relative;overflow:hidden}} .wf-bar span{{display:block;height:100%;background:#5b2be0;border-radius:6px}}
    .wf-meta{{width:120px;text-align:right;color:#8a90a2;flex:none}}
    .wf2{{border:1px solid #eef1f6;border-radius:10px;overflow:hidden;margin-top:4px}}
    .wf-row2{{display:grid;grid-template-columns:54px 200px 56px 56px 64px 1fr;gap:8px;align-items:center;font-size:11.5px;padding:5px 10px;border-bottom:1px solid #f4f5f8}}
    .wf-row2.wf-h2{{font-weight:700;color:#8a90a2;font-size:10px;background:#fafbfc;text-transform:uppercase}}
    .wf-row2.slow{{background:#fff5f5}}
    .wf-c-type{{color:#8a90a2;text-transform:uppercase;font-size:10px}}
    .wf-c-url{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,Consolas,monospace}}
    .wf-c-proto{{color:#8a90a2}} .wf-c-size,.wf-c-dur{{text-align:right}}
    .wf-c-track{{position:relative;height:13px}}
    .wf-seg{{position:absolute;top:0;height:13px;display:inline-flex;border-radius:3px;overflow:hidden}} .wf-seg span{{display:block;height:13px}}
    .wf-axis{{display:grid;grid-template-columns:54px 200px 56px 56px 64px 1fr;gap:8px;padding:1px 10px;border-bottom:1px solid #f4f5f8}}
    .wf-axis-r{{grid-column:6;display:flex;justify-content:space-between;font-size:9px;color:#b0b4c2}} .wf-axis-r i{{font-style:normal}}
    .wf-legend{{display:flex;gap:13px;flex-wrap:wrap;margin-top:7px;font-size:10.5px;color:#8a90a2}} .wf-legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}}
    .rp-wrap{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}}
    .rp-card{{background:#f8f9fd;border-radius:12px;padding:14px 16px}}
    .rp-h{{font-size:12px;font-weight:700;color:#5b2be0;text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px}}
    .rp-sub{{font-size:11px;color:#8a90a2;line-height:1.45;margin-bottom:10px}}
    .rep-stage{{background:#0f1020;border-radius:12px;padding:10px;text-align:center;min-height:120px}}
    .rep-img{{max-width:100%;max-height:520px;border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.3)}}
    .rep-ctl{{display:flex;align-items:center;gap:10px;margin-top:12px}}
    .rep-ctl button{{border:1px solid #d5d8e0;background:#fff;border-radius:8px;padding:7px 16px;font-size:13px;font-weight:600;cursor:pointer;color:#5b2be0}}
    .rep-ctl button:hover{{background:#f3f0ff}}
    .rep-pos{{font-size:12px;color:#8a90a2;font-variant-numeric:tabular-nums}}
    .rep-cap{{font-size:13px;font-weight:600;color:#1a1a2e;margin-left:auto}}
    .ai-remind{{background:#faf7ff;border:1px dashed #cbb6f0;color:#5b2be0;border-radius:12px;padding:12px 16px;margin-bottom:18px;font-size:13px}}
    .ai-block{{background:#faf7ff;border:1px solid #eee0fb;border-radius:12px;padding:12px 14px;margin-bottom:14px}}
    .ai-row{{display:flex;align-items:center;gap:10px;margin-bottom:6px;flex-wrap:wrap}}
    .ai-badge{{background:linear-gradient(135deg,#8b2fd6,#c5267f);color:#fff;font-size:11px;font-weight:700;padding:4px 12px;border-radius:999px;white-space:nowrap}}
    .ai-sum{{font-size:12px;color:#8a90a2;flex:1;min-width:160px}}
    .ai-get{{border:0;background:linear-gradient(135deg,#8b2fd6,#c5267f);color:#fff;font-size:12.5px;font-weight:700;padding:7px 16px;border-radius:999px;cursor:pointer;white-space:nowrap}}
    .ai-get:hover{{filter:brightness(1.07)}}
    .ai-get:disabled{{opacity:.6;cursor:default}}
    .ai-out{{margin-top:6px}}
    .ai-err{{font-size:12.5px;color:#c5221f;background:#fff5f5;border:1px solid #ffd7d7;border-radius:10px;padding:10px 14px}}
    .ai-text{{font-size:14px;line-height:1.6;color:#2a2a3e;background:#faf7ff;border:1px solid #eee0fb;border-radius:10px;padding:12px 14px;margin:0 0 6px}}
    .ai-extra{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:4px 0 6px}}
    .ai-sub{{background:#fbfbfe;border:1px solid #eef0f3;border-radius:10px;padding:10px 14px;font-size:12.5px}}
    .ai-sub b{{display:block;margin-bottom:4px;color:#5b2be0;font-size:12px}}
    .ai-sub ul,.ai-sub ol{{margin:0;padding-left:18px;line-height:1.5}}
    @media(max-width:680px){{.ai-extra{{grid-template-columns:1fr}}}}
    .tm-row{{display:grid;grid-template-columns:74px 1fr 80px;gap:8px;align-items:center;font-size:11.5px;margin-bottom:6px}}
    .tm-l{{color:#8a90a2}} .tm-v{{text-align:right;color:#3c4043;font-weight:600}}
    .tm-bar{{background:#e7e9f0;border-radius:5px;height:11px;overflow:hidden}} .tm-bar span{{display:block;height:11px;border-radius:5px}}
    @media(max-width:680px){{.rp-wrap{{grid-template-columns:1fr}}}}
    .foot{{text-align:center;color:#b0b4c2;font-size:12px;margin-top:24px}}
    .gloss{{display:grid;grid-template-columns:1fr 1fr;gap:10px 24px}}
    .gloss div{{font-size:12.5px;line-height:1.4}} .gloss b{{color:#5b2be0;display:block;margin-bottom:1px}} .gloss span{{color:#5a6072}}
    @media(max-width:680px){{.gloss{{grid-template-columns:1fr}}}}
    @media print{{body{{background:#fff}} .hero,.tcard,.kpi,.panel{{box-shadow:none;border:1px solid #eef0f3}} .tcard{{break-inside:avoid}}}}
    @media(max-width:760px){{.kpis{{grid-template-columns:repeat(2,1fr)}}.tiles{{grid-template-columns:repeat(2,1fr)}}}}
  </style></head><body><div class="wrap">
    {build_replay_carousel(pages)}
    <div class="hero">
      <div class="hero-gauge">{score_ring(avg_score, 116)}</div>
      <div><h1>🧭 {esc(flow_name)}</h1>
        <div class="meta">{len(pages)} transactions · {esc(device)} · {datetime.now().strftime("%c")}</div>
        <div class="verdict">{v}</div></div>
    </div>
    <div class="kpis">
      {kpi('Avg score', avg_score if avg_score is not None else '—', v.split(' —')[0], score_color(avg_score))}
      {kpi('Transactions', len(pages), str(len(ok)) + ' passed')}
      {kpi('Avg LCP', (str(avg_lcp) + ' ms') if avg_lcp is not None else '—', 'load speed')}
      {kpi('Requests', total_req or '—', str(total_kb) + ' KB total')}
      {kpi('Console errors', total_err, 'needs attention' if total_err else 'none', '#c5221f' if total_err else '#0c8a3e')}
    </div>
    {detail_html}
    <div class="panel"><h2>📋 Summary — all transactions</h2>
      <table><thead><tr><th>#</th><th>Transaction</th><th>Score</th><th>LCP</th><th>FCP</th><th>CLS</th><th>TTFB</th><th>DCL</th><th>Load</th><th>Interact</th><th>Req</th><th>Transfer</th></tr></thead><tbody>{sum_rows_html}</tbody></table>
      <p class="muted" style="margin:8px 0 0;font-size:12px">Single-user measurement. <b>LCP/FCP/CLS</b> = how it looked to the visitor · <b>TTFB</b> = server response · <b>DCL/Load</b> = when the page was ready · <b>Interact</b> = input responsiveness (INP) · <b>Req/Transfer</b> = network cost.</p>
    </div>
    <div class="panel" id="glossary">
      <h2>📖 What these terms mean <span class="muted" style="font-weight:400;font-size:13px">— for anyone reading this report</span></h2>
      <div class="gloss">
        <div><b>Score (0–100)</b><span>Overall speed grade. 90+ is good (green), 50–89 needs work (amber), under 50 is poor (red).</span></div>
        <div><b>LCP — Largest Contentful Paint</b><span>When the biggest thing on screen finished loading. Good ≤ 2.5s.</span></div>
        <div><b>FCP — First Contentful Paint</b><span>When the first text or image appeared. Good ≤ 1.8s.</span></div>
        <div><b>CLS — Layout shift</b><span>How much the page jumped around while loading. 0 = nothing moved (best).</span></div>
        <div><b>TTFB — Time to First Byte</b><span>How long the server took to start replying. The server's part of the wait.</span></div>
        <div><b>DCL / Load</b><span>When the page structure was ready, and when everything finished loading.</span></div>
        <div><b>Interact (INP)</b><span>How quickly the page responded when the user clicked or typed.</span></div>
        <div><b>Perceived response time</b><span>The real wait a single user feels = server response + the browser drawing the page.</span></div>
        <div><b>Waterfall</b><span>Every file the page downloaded, shown in order with how long each took. The longest bar held the page up most.</span></div>
      </div>
    </div>
    <div class="foot">Generated by <b>PulseLab</b> · 🔦 Lighthouse (lab) + 🛠️ Chrome DevTools/CDP (live) · single-user performance report.</div>
  </div>
  <script>
  // Per-page AI insights, generated on demand. Clicking the button asks the
  // running PulseLab server (/api/page-insights) for a plain-English LLM summary
  // of THIS page's audit data, then renders it inline. No LLM is called until the
  // reader clicks. Requires the PulseLab server to be running (works in the live
  // report; a detached copy with the server stopped shows a clear error).
  function escHtml(s){{var d=document.createElement('div');d.textContent=s==null?'':String(s);return d.innerHTML;}}
  function renderInsights(out, a){{
    var html='<p class="ai-text">'+escHtml(a.plainEnglish||a.summary||'')+'</p>';
    var extra='';
    if(a.topIssues&&a.topIssues.length){{
      extra+='<div class="ai-sub"><b>⚠️ Top issues</b><ul>'+a.topIssues.slice(0,4).map(function(t){{return '<li>'+escHtml(t)+'</li>';}}).join('')+'</ul></div>';
    }}
    if(a.recommendations&&a.recommendations.length){{
      extra+='<div class="ai-sub"><b>✅ Recommended fixes</b><ol>'+a.recommendations.slice(0,4).map(function(t){{return '<li>'+escHtml(t)+'</li>';}}).join('')+'</ol></div>';
    }}
    if(extra) html+='<div class="ai-extra">'+extra+'</div>';
    out.innerHTML=html; out.style.display='block';
  }}
  function getPageInsights(btn){{
    var block=btn.closest('.ai-block');
    var out=block.querySelector('.ai-out');
    var digest;
    try{{ digest=JSON.parse(block.getAttribute('data-ai-digest')); }}
    catch(e){{ out.innerHTML='<div class="ai-err">Could not read this page\\'s data.</div>'; out.style.display='block'; return; }}
    btn.disabled=true; var orig=btn.textContent; btn.textContent='✨ Thinking…';
    out.style.display='block'; out.innerHTML='<p class="ai-text">Generating insights…</p>';
    fetch('/api/page-insights',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{digest:digest}})}})
      .then(function(r){{return r.json().then(function(j){{return {{ok:r.ok,j:j}};}});}})
      .then(function(res){{
        if(res.ok&&res.j&&res.j.analysis){{ renderInsights(out,res.j.analysis); btn.style.display='none'; }}
        else{{ out.innerHTML='<div class="ai-err">'+escHtml((res.j&&res.j.error)||'Could not generate insights.')+'</div>'; btn.disabled=false; btn.textContent=orig; }}
      }})
      .catch(function(){{
        out.innerHTML='<div class="ai-err">Couldn\\'t reach the PulseLab server. AI insights need the server running (open this report from a live PulseLab run).</div>';
        btn.disabled=false; btn.textContent=orig;
      }});
  }}
  </script>
  </body></html>"""


def score_color_band(band):
    """Map a band dict/string to a colour for the CDP tiles."""
    level = band.get("level") if isinstance(band, dict) else band
    return {"good": "#0c8a3e", "average": "#b8860b", "poor": "#c5221f"}.get(level, "#9aa0a6")


def band_text(band):
    level = band.get("level") if isinstance(band, dict) else band
    return {"good": "Good", "average": "Needs work", "poor": "Poor"}.get(level, "")


def band_tile(label, val, band):
    color = score_color_band(band)
    txt = band_text(band)
    tr = f'<div class="tr" style="color:{color}">{txt}</div>' if txt else ""
    return (f'<div class="tile" style="--bc:{color}"><div class="tl">{label}</div>'
            f'<div class="tv">{val if val is not None else "n/a"}</div>{tr}</div>')


def build_journey_markdown(flow_name, pages, device):
    """Markdown version of the overall journey report."""
    ok = [p for p in pages if not p.get("error") and p.get("performanceScore") is not None]
    avg = round(sum(p["performanceScore"] for p in ok) / len(ok)) if ok else "—"
    lines = [
        f"# 🧭 Journey report — {flow_name}", "",
        f"{len(pages)} transactions · device: {device} · {datetime.now().strftime('%c')}", "",
        f"**Average performance score: {avg}**", "",
        "## Summary", "",
        "| # | Transaction | Score | LCP | FCP | CLS | TBT | Transferred |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, p in enumerate(pages):
        if p.get("error"):
            lines.append(f"| {i+1} | {p['name']} | ERROR | | | | | |")
            continue
        lines.append(
            f"| {i+1} | {p['name']} | {p.get('performanceScore', '—')} | "
            f"{_metric_val(p, 'lcp')} | {_metric_val(p, 'fcp')} | {_metric_val(p, 'cls')} | "
            f"{_metric_val(p, 'tbt')} | {(p.get('cdp') or {}).get('transferKb', '—')} KB |"
        )
    for i, p in enumerate(pages):
        lines += ["", f"## {i+1}. {p['name']}", f"`{p.get('url')}`", ""]
        if p.get("error"):
            lines.append(f"Error: {p['error']}")
            continue
        lines.append("**Lighthouse (lab):** " + " · ".join(
            f"{m['label']} {m['value']} ({m.get('score', '—')})" for m in (p.get("metrics") or [])
        ))
        cdp = p.get("cdp")
        if cdp and cdp.get("metrics"):
            c = cdp["metrics"]
            lines += ["", (
                f"**CDP (live):** FCP {c.get('fcp', 'n/a')}ms · LCP {c.get('lcp', 'n/a')}ms · "
                f"CLS {c.get('cls', 'n/a')} · TTFB {c.get('ttfb', 'n/a')}ms · Load {c.get('load', 'n/a')}ms · "
                f"{cdp.get('totalRequests', '—')} requests · {cdp.get('transferKb', '—')} KB · "
                f"{cdp.get('consoleErrors')} console errors"
            )]
        if p.get("opportunities"):
            lines += ["", "**Opportunities:** " + "; ".join(
                f"{o['title']} (~{o.get('saveMs')}ms)" for o in p["opportunities"]
            )]
    return "\n".join(lines)


def summarise_page(name, url, lhr, thresholds, report_url=None, json_url=None, cdp=None, lighthouse_skipped=False):
    # If Lighthouse got redirected to login (protected page it can't cold-load),
    # its numbers describe the login page, not the real one — so we drop them and
    # rely on the CDP (live, logged-in) capture, marking Lighthouse N/A honestly.
    if lighthouse_skipped:
        return {
            "name": name, "url": url,
            "finalUrl": (cdp or {}).get("finalUrl") or url,
            "performanceScore": None,
            "verdict": "Lighthouse n/a — page needs login (measured live via CDP instead)",
            "lighthouseSkipped": True,
            "categories": [], "metrics": [], "opportunities": [],
            "insights": ["This page requires a logged-in session, which Lighthouse cannot cold-load. The live CDP capture below reflects the real page."],
            "aiAnalysis": build_llm_analysis(name, url, None, "login-gated (CDP only)", [], [], cdp),
            "cdp": cdp, "reportUrl": None, "jsonUrl": None,
        }
    metrics = extract_metrics(lhr)
    categories = extract_categories(lhr)
    perf = next((c for c in categories if c["key"] == "performance"), None)
    score = perf["score"] if perf else None
    verdict_text = verdict(score)
    opportunities = extract_opportunities(lhr)
    return {
        "name": name, "url": url,
        "finalUrl": lhr.get("finalDisplayedUrl") or lhr.get("finalUrl") or url,
        "performanceScore": score,
        "verdict": verdict_text,
        "categories": categories,
        "metrics": metrics,
        "opportunities": opportunities,
        "insights": build_insights(metrics, lhr),  # rule-based, always present
        "aiAnalysis": build_llm_analysis(name, url, score, verdict_text, metrics, opportunities, cdp),  # LLM, or None
        "cdp": cdp,
        "reportUrl": report_url,
        "jsonUrl": json_url,
    }


def metric_numeric(page, key):
    for m in page.get("metrics") or []:
        if m["key"] == key:
            return m.get("numeric")
    return None


def record_run(run):
    """Stamp a stable unique id so the UI can delete / compare a specific run."""
    run["id"] = f"{int(time.time() * 1000):x}-{uuid.uuid4().hex[:6]}"
    history = read_json(HISTORY_FILE, [])
    history.insert(0, run)
    write_json(HISTORY_FILE, history[:200])


def url_to_disk_path(url):
    """Map a /reports/... or /downloads/... URL back to a file on disk."""
    if not url or not isinstance(url, str):
        return None
    m = re.match(r"^/(reports|downloads)/(.+)$", url)
    if not m:
        return None
    base = REPORTS_DIR if m.group(1) == "reports" else DATA_DIR
    p = os.path.join(base, os.path.basename(m.group(2)))  # basename guards traversal
    return p if os.path.abspath(p).startswith(os.path.abspath(base)) else None


def run_file_urls(run):
    """Collect every report/download URL a run produced, so deleting a run also
    removes its files from disk (no orphaned reports)."""
    urls = set()
    if run.get("reportUrl"):
        urls.add(run["reportUrl"])
    for f in (run.get("formats") or {}).values():
        if f:
            urls.add(f)
    for r in run.get("reports") or []:
        if r.get("url"):
            urls.add(r["url"])
    return list(urls)


def list_flows():
    """Discover saved journeys from the flows/ folder (.py scripted, or .json)."""
    if not os.path.exists(FLOWS_DIR):
        return []
    flows = []
    for f in sorted(os.listdir(FLOWS_DIR)):
        ext = os.path.splitext(f)[1].lower()
        if ext not in (".py", ".json"):
            continue
        if ext == ".py" and (f.startswith("_") or f == "__init__.py"):
            continue
        flow = {"name": os.path.splitext(f)[0], "file": f, "type": ext[1:].upper(),
                "steps": None, "startUrl": None}
        if ext == ".json":
            j = read_json(os.path.join(FLOWS_DIR, f), {})
            steps = j.get("steps") if isinstance(j, dict) and isinstance(j.get("steps"), list) else (j if isinstance(j, list) else [])
            flow["steps"] = len(steps) or None
            flow["startUrl"] = (j.get("startUrl") or j.get("start") if isinstance(j, dict) else None) or (steps[0].get("url") if steps else None)
        else:
            # .py: count AUDIT_POINTS / surface the first URL without executing.
            pts = _peek_py_audit_points(os.path.join(FLOWS_DIR, f))
            flow["steps"] = len(pts) or None
            flow["startUrl"] = pts[0]["url"] if pts else None
        flows.append(flow)
    return flows


# -----------------------------------------------------------------------------
#  Loading flow files (100% Python — no Node).
#    .py   : a module exposing run(page, context, log) + AUDIT_POINTS, with an
#            optional setup(page, context). Loaded via importlib and executed
#            in-process against the Playwright page we control.
#    .json : { startUrl, steps:[{name,url}] } (or a bare [{url}] list) — parsed
#            directly, no script to run.
# -----------------------------------------------------------------------------
def resolve_audit_points_json(flow_path):
    j = read_json(flow_path, {})
    steps = j.get("steps") if isinstance(j, dict) and isinstance(j.get("steps"), list) else (j if isinstance(j, list) else [])
    points = [{"name": s.get("name") or f"Page {i+1}", "url": s["url"]}
              for i, s in enumerate(steps) if s and s.get("url")]
    if isinstance(j, dict) and j.get("startUrl"):
        points.insert(0, {"name": "Start", "url": j["startUrl"]})
    if not points:
        start = j.get("startUrl") or j.get("start") if isinstance(j, dict) else None
        points = [{"name": "start", "url": start}]
    return points


def load_py_flow(flow_path):
    """Import a .py flow module fresh (no caching, so edits take effect) and
    return it. The module must expose run(page, context, log) + AUDIT_POINTS;
    setup(page, context) is optional."""
    mod_name = f"_flow_{os.path.splitext(os.path.basename(flow_path))[0]}_{uuid.uuid4().hex[:6]}"
    spec = importlib.util.spec_from_file_location(mod_name, flow_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Could not load flow module: {flow_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run") or not callable(mod.run):
        raise RuntimeError("Python flow must define run(page, context, log)")
    return mod


def _peek_py_audit_points(flow_path):
    """Read AUDIT_POINTS from a .py flow WITHOUT running it (used for the flow
    list + preview). Tries a clean import; falls back to a regex scrape if the
    module imports something unavailable."""
    try:
        mod = load_py_flow(flow_path)
        pts = getattr(mod, "AUDIT_POINTS", None)
        if isinstance(pts, list):
            return [{"name": p.get("name"), "url": p.get("url")} for p in pts if p.get("url")]
    except Exception:
        pass
    try:
        with open(flow_path, "r", encoding="utf-8") as fh:
            src = fh.read()
        m = re.search(r"AUDIT_POINTS\s*=\s*(\[[\s\S]*?\])", src)
        if not m:
            return []
        names = re.findall(r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]", m.group(1))
        urls = re.findall(r"['\"]url['\"]\s*:\s*['\"]([^'\"]+)['\"]", m.group(1))
        return [{"name": n, "url": u} for n, u in zip(names, urls)]
    except Exception:
        return []


# =============================================================================
#  HTTP ROUTES (Flask)
# =============================================================================
app = Flask(__name__, static_folder=None)

# Quiet the per-request access log for the high-frequency polling endpoints
# (the UI hits /api/live and /api/state every ~1s). They'd otherwise flood the
# terminal; real actions (audits, traversals, errors) still log normally.
class _QuietPolling(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "/api/live" not in msg and "/api/state" not in msg


logging.getLogger("werkzeug").addFilter(_QuietPolling())


@app.get("/")
def index():
    return send_file(os.path.join(PUBLIC_DIR, "index.html"))


@app.get("/api/state")
def api_state():
    return jsonify({
        "ready": not audit_running,
        "running": audit_running,
        "thresholds": read_thresholds(),
        "flows": list_flows(),
        "history": read_json(HISTORY_FILE, [])[:50],
        "llm": {"enabled": llm_enabled(), "model": llm_config()["deployment"] if llm_enabled() else None},
    })


@app.get("/api/thresholds")
def api_get_thresholds():
    return jsonify({"thresholds": read_thresholds()})


@app.post("/api/thresholds")
def api_save_thresholds():
    incoming = (request.json or {}).get("thresholds")
    if not incoming or not isinstance(incoming, dict):
        return jsonify({"error": "Missing thresholds object"}), 400
    current = read_thresholds()
    for key in current:
        if incoming.get(key, {}).get("good") is not None:
            current[key]["good"] = float(incoming[key]["good"])
        if incoming.get(key, {}).get("ni") is not None:
            current[key]["ni"] = float(incoming[key]["ni"])
    write_json(THRESHOLDS_FILE, current)
    return jsonify({"message": "Thresholds saved", "thresholds": current})


@app.get("/api/history")
def api_history():
    return jsonify({"history": read_json(HISTORY_FILE, [])})


@app.delete("/api/history/<run_id>")
def api_delete_history(run_id):
    history = read_json(HISTORY_FILE, [])
    idx = next((i for i, r in enumerate(history) if r.get("id") == run_id), -1)
    if idx == -1:
        return jsonify({"message": "Run not found"}), 404
    run = history.pop(idx)
    removed = 0
    for url in run_file_urls(run):
        p = url_to_disk_path(url)
        if p and os.path.exists(p):
            try:
                os.remove(p)
                removed += 1
            except Exception:
                pass
    write_json(HISTORY_FILE, history)
    return jsonify({"message": "Run deleted", "filesRemoved": removed})


@app.get("/api/compare")
def api_compare():
    ids = [s.strip() for s in (request.args.get("ids") or "").split(",") if s.strip()]
    if len(ids) < 2:
        return jsonify({"message": "Pick at least 2 runs to compare"}), 400
    history = read_json(HISTORY_FILE, [])
    by_id = {r.get("id"): r for r in history}
    runs = [{
        "id": r["id"], "name": r.get("name"), "kind": r.get("kind"), "date": r.get("date"),
        "score": r.get("score"), "pages": r.get("pages"), "avgLcp": r.get("avgLcp"),
        "avgCls": r.get("avgCls"), "reportUrl": r.get("reportUrl"),
    } for r in (by_id.get(i) for i in ids) if r]
    if len(runs) < 2:
        return jsonify({"message": "Some runs were not found"}), 404
    return jsonify({"runs": runs})


@app.get("/api/compare/ai")
def api_compare_ai():
    """Plain-English AI comparison of 2+ runs (opt-in, on the Compare page)."""
    if not llm_enabled():
        return jsonify({"error": "LLM not configured. Add AZURE_OPENAI_* to pulselab/.env."}), 400
    ids = [s.strip() for s in (request.args.get("ids") or "").split(",") if s.strip()]
    if len(ids) < 2:
        return jsonify({"error": "Pick at least 2 runs to compare"}), 400
    history = read_json(HISTORY_FILE, [])
    by_id = {r.get("id"): r for r in history}
    runs = [{
        "name": r.get("name"), "date": r.get("date"), "score": r.get("score"),
        "pages": r.get("pages"), "avgLcp": r.get("avgLcp"), "avgCls": r.get("avgCls"),
    } for r in (by_id.get(i) for i in ids) if r]
    if len(runs) < 2:
        return jsonify({"error": "Some runs were not found"}), 404
    system = (
        "You compare web-performance test runs for a NON-TECHNICAL reader. Given 2+ runs "
        "with score, average LCP (load speed) and CLS (layout stability), say in plain English "
        "which run is better and why, whether performance improved or regressed, and the single "
        "most important takeaway. 3-4 sentences, plain prose, no markdown."
    )
    try:
        txt = llm_chat(system, "Runs (JSON):\n" + json_lib.dumps(runs, indent=2), max_tokens=400, temperature=0.3)
        return jsonify({"analysis": txt.strip()})
    except Exception as e:
        return jsonify({"error": "AI comparison failed: " + str(e)}), 502


@app.get("/api/live")
def api_live():
    return jsonify(LIVE)


@app.post("/api/flow")
def api_save_flow():
    body = request.json or {}
    name, content, ftype = body.get("name"), body.get("content"), body.get("type")
    if not content or not str(content).strip():
        return jsonify({"error": "Flow content is empty"}), 400
    stripped = str(content).strip()
    # .json for a pasted JSON page-list; otherwise a Python scripted flow (.py).
    ext = ftype if ftype in ("json", "py") else ("json" if stripped.startswith(("{", "[")) else "py")
    safe = re.sub(r"[^a-z0-9._-]", "-", str(name or "flow"), flags=re.I)
    safe = re.sub(r"\.(py|json)$", "", safe, flags=re.I)[:60] or "flow"
    if ext == "json":
        try:
            json_lib.loads(content)
        except Exception as e:
            return jsonify({"error": "Invalid JSON: " + str(e)}), 400
    file = f"{safe}.{ext}"
    with open(os.path.join(FLOWS_DIR, file), "w", encoding="utf-8") as fh:
        fh.write(content)
    return jsonify({"message": "Flow saved", "file": file, "flows": list_flows()})


# Few-shot contract handed to the LLM so its output matches PulseLab's runnable
# .py flow shape (run(page, context, log) + AUDIT_POINTS, Playwright sync API).
_FLOW_CONTRACT = '''import re

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
]'''


def _clean_py_from_llm(text):
    """Strip markdown fences / prose so we keep just the Python source."""
    t = text.strip()
    m = re.search(r"```(?:python)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    return t


@app.post("/api/flow/generate")
def api_flow_generate():
    """NLP -> runnable .py flow via the configured LLM (Azure OpenAI)."""
    if not llm_enabled():
        return jsonify({"error": "LLM not configured. Add AZURE_OPENAI_* keys to pulselab/.env, then restart."}), 400
    body = request.json or {}
    name, prompt = body.get("name"), body.get("prompt")
    if not prompt or not str(prompt).strip():
        return jsonify({"error": "Describe the journey first."}), 400

    system = (
        "You convert a plain-English web journey into a PulseLab flow file. "
        "Output ONLY valid Python (no markdown, no prose) using the Playwright SYNC API. "
        "Define run(page, context, log) that performs the journey with goto/fill/click and "
        "wait_for_load_state, and an AUDIT_POINTS list of the distinct pages to measure "
        "(each {\"name\": str, \"url\": str}). Add setup(page, context) only if login is needed. "
        "Use stable selectors (#id, [name], get_by_role). Here is the exact required shape:\n\n"
        + _FLOW_CONTRACT
    )
    try:
        raw = llm_chat(system, "Journey:\n" + str(prompt), max_tokens=1400, temperature=0.1)
    except Exception as e:
        return jsonify({"error": "LLM call failed: " + str(e)}), 502

    src = _clean_py_from_llm(raw)
    # Validate it compiles; one repair attempt, then give up with the error.
    try:
        ast.parse(src)
    except SyntaxError as e:
        try:
            fixed = llm_chat(
                system,
                f"This Python had a syntax error ({e}). Return a corrected, complete version, Python only:\n\n{src}",
                max_tokens=1400, temperature=0.0,
            )
            src = _clean_py_from_llm(fixed)
            ast.parse(src)
        except Exception as e2:
            return jsonify({"error": "Generated flow did not compile: " + str(e2),
                            "draft": src}), 422

    safe = re.sub(r"[^a-z0-9._-]", "-", str(name or "llm-flow"), flags=re.I)
    safe = re.sub(r"\.(py|json)$", "", safe, flags=re.I)[:60] or "llm-flow"
    file = f"{safe}.py"
    with open(os.path.join(FLOWS_DIR, file), "w", encoding="utf-8") as fh:
        fh.write(src)
    return jsonify({"message": "Flow generated", "file": file, "flows": list_flows()})


@app.post("/api/codegen/launch")
def api_codegen_launch():
    body = request.json or {}
    url = str(body.get("url","") or "").strip()
    if not url:
        return jsonify({"error": "Missing start URL"}), 400

    browser = body.get("browser") or "chromium"
    device = body.get("device") or ""
    auth_profile = str(body.get("authProfile") or "").strip()
    name = str(body.get("name") or "playwright-codegen").strip() or "playwright-codegen"
    save_auth = bool(body.get("saveAuth"))
    capture_har = bool(body.get("captureHar"))
    keep_py = bool(body.get("keepPy"))

    npx_path = _node_tool("npx")
    npm_path = _node_tool("npm")

    if npx_path:
        args = [npx_path, "playwright", "codegen", url, "--target", "python", "--browser", browser]
    elif npm_path:
        args = [npm_path, "exec", "--no-install", "playwright", "codegen", url, "--target", "python", "--browser", browser]
    else:
        return jsonify({
            "error": "npx/npm not found. Install Node.js, ensure npm is on PATH, or start the server from the same shell where npx/npm works."
        }), 500

    if device:
        args += ["--device", device]

    if save_auth or auth_profile:
        profile_name = auth_profile or name
        profile_name = re.sub(r"[^a-z0-9._-]", "-", profile_name, flags=re.I)
        profile_name = re.sub(r"\.(json|py)$", "", profile_name, flags=re.I)[:60] or "profile"
        storage_path = os.path.join(PROFILES_DIR, f"{profile_name}-storage-state.json")
        args += ["--save-storage", storage_path]

    if capture_har:
        har_name = re.sub(r"[^a-z0-9._-]", "-", name, flags=re.I)
        har_name = re.sub(r"\.(json|py)$", "", har_name, flags=re.I)[:60] or "session"
        har_path = os.path.join(DATA_DIR, f"{har_name}.har")
        args += ["--record-har", har_path]

    if keep_py:
        output_name = re.sub(r"[^a-z0-9._-]", "-", name, flags=re.I)
        output_name = re.sub(r"\.(json|py)$", "", output_name, flags=re.I)[:60] or "playwright-codegen"
        output_path = os.path.join(FLOWS_DIR, f"{output_name}.py")
        args += ["--output", output_path]

    try:
        subprocess.Popen(args, cwd=BASE_DIR, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return jsonify({"error": "npx not found. Install Node.js and add it to PATH."}), 500
    except Exception as e:
        return jsonify({"error": "Could not start codegen: " + str(e)}), 500

    return jsonify({"message": "Playwright codegen launched", "cmd": " ".join(args)})


@app.get("/api/flow/<file>")
def api_get_flow(file):
    fp = os.path.join(FLOWS_DIR, os.path.basename(file))
    if not os.path.exists(fp):
        return jsonify({"error": "Flow not found"}), 404
    with open(fp, "r", encoding="utf-8") as fh:
        return jsonify({"file": os.path.basename(fp), "content": fh.read()})


@app.delete("/api/flow/<file>")
def api_delete_flow(file):
    fp = os.path.join(FLOWS_DIR, os.path.basename(file))
    if not os.path.exists(fp):
        return jsonify({"error": "Flow not found"}), 404
    os.remove(fp)
    return jsonify({"message": "Flow deleted", "flows": list_flows()})


@app.get("/api/flow-preview/<file>")
def api_flow_preview(file):
    fp = os.path.join(FLOWS_DIR, os.path.basename(file))
    if not os.path.exists(fp):
        return jsonify({"error": "Flow not found"}), 404
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".py":
        steps = _peek_py_audit_points(fp)
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                src = fh.read()
            has_setup = bool(re.search(r"^def\s+setup\s*\(", src, re.M))
        except Exception:
            has_setup = False
        return jsonify({"file": os.path.basename(fp), "steps": steps, "hasSetup": has_setup})
    steps = resolve_audit_points_json(fp)
    return jsonify({"file": os.path.basename(fp), "steps": steps, "hasSetup": False})


@app.post("/api/flow/clone/<file>")
def api_flow_clone(file):
    """Duplicate a flow as <name>-copy.<ext> (used by the flow-detail Clone button)."""
    fp = os.path.join(FLOWS_DIR, os.path.basename(file))
    if not os.path.exists(fp):
        return jsonify({"error": "Flow not found"}), 404
    base, ext = os.path.splitext(os.path.basename(fp))
    new_name = f"{base}-copy{ext}"
    i = 2
    while os.path.exists(os.path.join(FLOWS_DIR, new_name)):
        new_name = f"{base}-copy{i}{ext}"
        i += 1
    with open(fp, "r", encoding="utf-8") as src, open(os.path.join(FLOWS_DIR, new_name), "w", encoding="utf-8") as dst:
        dst.write(src.read())
    return jsonify({"message": "Flow cloned", "file": new_name, "flows": list_flows()})


# --- RECORD a flow ----------------------------------------------------------
# Recording opens a long-lived visible browser. With Playwright's sync API that
# must live on its own thread; we keep the handle and the captured navigations.
recording_state = None
_recording_lock = threading.Lock()


# In-page recorder: injected into every page/frame. Listens for the user's
# clicks, typing and dropdown changes, builds a stable CSS selector for each
# target (prefers #id, then [name], then data-test/aria-label, then a short
# nth-of-type path), and reports the event to Python via the exposed binding.
RECORDER_JS = r"""
(() => {
  if (window.__pulselabRecorderInstalled) return;
  window.__pulselabRecorderInstalled = true;

  const cssEscape = (s) => (window.CSS && CSS.escape) ? CSS.escape(s) : String(s).replace(/([^a-zA-Z0-9_-])/g, '\\$1');

  function selectorFor(el) {
    if (!el || el.nodeType !== 1) return null;
    if (el.id) return '#' + cssEscape(el.id);
    for (const a of ['data-test', 'data-testid', 'name']) {
      const v = el.getAttribute && el.getAttribute(a);
      if (v) return el.tagName.toLowerCase() + '[' + a + '="' + v + '"]';
    }
    const al = el.getAttribute && el.getAttribute('aria-label');
    if (al) return el.tagName.toLowerCase() + '[aria-label="' + al + '"]';
    // Fallback: a short structural path with :nth-of-type.
    const parts = [];
    let node = el;
    while (node && node.nodeType === 1 && parts.length < 4) {
      let part = node.tagName.toLowerCase();
      if (node.id) { parts.unshift('#' + cssEscape(node.id)); break; }
      const parent = node.parentElement;
      if (parent) {
        const sibs = [...parent.children].filter(c => c.tagName === node.tagName);
        if (sibs.length > 1) part += ':nth-of-type(' + (sibs.indexOf(node) + 1) + ')';
      }
      parts.unshift(part);
      node = node.parentElement;
    }
    return parts.join(' > ');
  }

  const send = (ev) => { try { window.__pulselabRecord(ev); } catch (e) {} };
  const isTextInput = (el) =>
    el && (el.tagName === 'TEXTAREA' ||
      (el.tagName === 'INPUT' && !['checkbox','radio','button','submit','file'].includes((el.type||'').toLowerCase())));

  document.addEventListener('click', (e) => {
    const el = e.target.closest('a,button,input[type=submit],input[type=button],[role=button],[onclick]') || e.target;
    if (!el) return;
    if (isTextInput(el)) return;  // typing is captured on input, not click
    send({ action: 'click', selector: selectorFor(el),
           text: (el.innerText || el.value || '').replace(/\s+/g, ' ').trim().slice(0, 40), url: location.href });
  }, true);

  document.addEventListener('input', (e) => {
    const el = e.target;
    if (!isTextInput(el)) return;
    send({ action: 'fill', selector: selectorFor(el), value: el.value, url: location.href });
  }, true);

  document.addEventListener('change', (e) => {
    const el = e.target;
    if (el && el.tagName === 'SELECT') {
      send({ action: 'select', selector: selectorFor(el), value: el.value, url: location.href });
    }
  }, true);
})();
"""


def _py_str(s):
    """Safely embed a recorded string into generated Python source."""
    return '"' + str(s or "").replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_py_flow_from_actions(name, start_url, actions, visited):
    """Turn recorded navigations + interactions into a runnable .py flow:
    run(page, context, log) with goto/fill/click/select_option + AUDIT_POINTS."""
    lines = [
        '"""',
        f"{name} — recorded by PulseLab (Record it live).",
        "",
        "Captured live in a real browser: each click, field entry and dropdown",
        "is replayed as a Playwright step. Edit freely — it is normal Python.",
        '"""',
        "",
        "",
        "def run(page, context, log):",
        f'    log("opening {start_url}")',
        f'    page.goto({_py_str(start_url)}, wait_until="domcontentloaded")',
    ]
    for a in actions:
        sel = a.get("selector")
        if not sel:
            continue
        act = a.get("action")
        if act == "fill":
            lines.append(f'    log("fill {sel}")')
            lines.append(f'    page.locator({_py_str(sel)}).fill({_py_str(a.get("value",""))})')
        elif act == "select":
            lines.append(f'    log("select {sel}")')
            lines.append(f'    page.locator({_py_str(sel)}).select_option({_py_str(a.get("value",""))})')
        elif act == "click":
            label = re.sub(r"\s+", " ", str(a.get("text") or sel)).strip()[:40]
            lines.append(f'    log({_py_str("click " + label)})')
            lines.append(f'    page.locator({_py_str(sel)}).first.click()')
            lines.append('    page.wait_for_load_state("domcontentloaded")')
    lines.append('    log("recorded flow complete")')
    lines.append("")
    lines.append("")
    # AUDIT_POINTS: the distinct pages the journey visited.
    pts = [s for i, s in enumerate(visited) if i == 0 or s["url"] != visited[i - 1]["url"]]
    if not pts:
        pts = [{"name": "start", "url": start_url}]
    lines.append("AUDIT_POINTS = [")
    for i, p in enumerate(pts):
        nm = p.get("name") or f"step-{i+1}"
        lines.append(f'    {{"name": {_py_str(nm)}, "url": {_py_str(p["url"])}}},')
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


@app.post("/api/record/start")
def api_record_start():
    global recording_state
    body = request.json or {}
    url, name = body.get("url"), body.get("name")
    if not url:
        return jsonify({"error": "Missing start URL"}), 400
    if recording_state:
        return jsonify({"error": "A recording is already in progress."}), 409

    visited = []          # page navigations (for AUDIT_POINTS)
    actions = []          # every user interaction (click / fill / select) in order
    ready = threading.Event()
    stop_event = threading.Event()
    err_box = {}

    def record_thread():
        global recording_state
        try:
            with sync_playwright() as pw:
                browser = launch_browser(pw, [], "auto", headed=True)
                context = browser.new_context(no_viewport=True)

                # Receive interaction events from the in-page recorder script. The
                # browser-side script (RECORDER_JS) computes a stable selector for
                # each target and reports {action, selector, value, url} here.
                def on_action(source, ev):
                    try:
                        if not isinstance(ev, dict) or not ev.get("action"):
                            return
                        # Collapse repeated keystrokes in the same field into one fill.
                        if (ev["action"] == "fill" and actions
                                and actions[-1].get("action") == "fill"
                                and actions[-1].get("selector") == ev.get("selector")):
                            actions[-1]["value"] = ev.get("value", "")
                        else:
                            actions.append(ev)
                    except Exception:
                        pass

                context.expose_binding("__pulselabRecord", on_action)
                context.add_init_script(RECORDER_JS)

                page = context.new_page()

                def on_nav(frame):
                    if frame == page.main_frame:
                        u = frame.url
                        if u and u != "about:blank" and (not visited or visited[-1]["url"] != u):
                            visited.append({"name": f"step-{len(visited)+1}", "url": u})

                page.on("framenavigated", on_nav)
                try:
                    page.goto(url, wait_until="domcontentloaded")
                except Exception:
                    pass
                recording_state = {"visited": visited, "actions": actions,
                                   "name": name or "recorded-flow",
                                   "page": page, "stop": stop_event}
                ready.set()
                # IMPORTANT: pump Playwright's event loop while idle so framenavigated
                # and the exposed binding actually fire. A bare time.sleep() does NOT
                # pump the driver, which is why earlier recordings only ever caught the
                # first page. wait_for_timeout() is a Playwright call, so events flow.
                while not stop_event.is_set():
                    try:
                        page.wait_for_timeout(300)
                    except Exception:
                        break
                try:
                    browser.close()
                except Exception:
                    pass
        except Exception as e:
            err_box["err"] = str(e)
            ready.set()

    threading.Thread(target=record_thread, daemon=True).start()
    ready.wait(timeout=30)
    if err_box.get("err"):
        recording_state = None
        return jsonify({"error": "Could not open browser: " + err_box["err"]}), 500
    return jsonify({"message": "Recording started — click through the journey in the opened browser, then Stop.",
                    "startUrl": url})


@app.get("/api/record/status")
def api_record_status():
    if not recording_state:
        return jsonify({"recording": False})
    try:
        current = recording_state["page"].url
    except Exception:
        current = None
    return jsonify({"recording": True, "steps": len(recording_state["visited"]),
                    "actions": len(recording_state.get("actions", [])),
                    "current": current, "name": recording_state["name"]})


@app.post("/api/record/stop")
def api_record_stop():
    global recording_state
    if not recording_state:
        return jsonify({"error": "No recording in progress."}), 400
    state = recording_state
    visited = list(state["visited"])
    actions = list(state.get("actions", []))
    name = state["name"]
    state["stop"].set()  # let the recording thread close the browser
    recording_state = None
    time.sleep(0.5)
    steps = [s for i, s in enumerate(visited) if i == 0 or s["url"] != visited[i - 1]["url"]]
    if not steps and not actions:
        return jsonify({"error": "Nothing was recorded — navigate or interact, then Stop."}), 400

    safe = re.sub(r"[^a-z0-9._-]", "-", str(name), flags=re.I)
    safe = re.sub(r"\.(json|py)$", "", safe, flags=re.I)[:60] or "recorded-flow"
    start_url = steps[0]["url"] if steps else (actions[0].get("url") if actions else "")

    if actions:
        # Full interaction capture → a runnable .py flow (clicks, fills, selects).
        file = f"{safe}.py"
        src = build_py_flow_from_actions(name, start_url, actions, visited)
        with open(os.path.join(FLOWS_DIR, file), "w", encoding="utf-8") as fh:
            fh.write(src)
        return jsonify({"message": "Recording saved", "file": file,
                        "steps": len(actions), "flows": list_flows()})

    # No interactions captured — fall back to the URL-only .json page list.
    file = f"{safe}.json"
    flow = {"name": name, "startUrl": steps[0]["url"], "steps": steps, "recordedAt": now_iso()}
    write_json(os.path.join(FLOWS_DIR, file), flow)
    return jsonify({"message": "Recording saved", "file": file, "steps": len(steps), "flows": list_flows()})


# =============================================================================
#  AUTH PROFILES — saved Playwright storage_state, so journeys skip SSO/login.
#  A "profile" is a storage_state.json (cookies + localStorage) captured once
#  from a logged-in browser, then replayed on every audit.
# =============================================================================
def _profile_path(name):
    safe = re.sub(r"[^a-z0-9._-]", "-", str(name), flags=re.I)
    safe = re.sub(r"\.json$", "", safe, flags=re.I)[:80] or "profile"
    return os.path.join(PROFILES_DIR, f"{safe}.json"), safe


def list_profiles():
    """Saved storage_state profiles: name, size, when saved."""
    out = []
    if os.path.exists(PROFILES_DIR):
        for f in sorted(os.listdir(PROFILES_DIR)):
            if not f.endswith(".json"):
                continue
            p = os.path.join(PROFILES_DIR, f)
            try:
                st = os.stat(p)
                out.append({
                    "name": os.path.splitext(f)[0],
                    "file": f,
                    "sizeKb": round(st.st_size / 1024, 1),
                    "savedAt": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })
            except Exception:
                pass
    return out


@app.get("/api/profiles")
def api_profiles():
    return jsonify({"profiles": list_profiles()})


@app.delete("/api/profiles/<name>")
def api_delete_profile(name):
    p, _ = _profile_path(name)
    if not os.path.exists(p):
        return jsonify({"error": "Profile not found"}), 404
    os.remove(p)
    return jsonify({"message": "Profile deleted", "profiles": list_profiles()})


# --- RECORD A NEW PROFILE (headed browser: log in, then close the window) ----
# Opens a visible browser at the login URL. The user signs in, then clicks Stop
# (or closes the page); we capture context.storage_state() and save it.
profile_rec_state = None


@app.post("/api/profile/record/start")
def api_profile_record_start():
    global profile_rec_state
    body = request.json or {}
    name, url = body.get("name"), body.get("url")
    if not name or not url:
        return jsonify({"error": "Profile name and login URL are both required"}), 400
    if profile_rec_state:
        return jsonify({"error": "A profile recording is already in progress."}), 409

    ready = threading.Event()
    stop_event = threading.Event()
    result_box = {}

    def rec_thread():
        global profile_rec_state
        try:
            with sync_playwright() as pw:
                browser = launch_browser(pw, [], "auto", headed=True)
                context = browser.new_context(no_viewport=True)
                page = context.new_page()
                try:
                    page.goto(url, wait_until="domcontentloaded")
                except Exception:
                    pass
                profile_rec_state = {"page": page, "name": name, "stop": stop_event}
                ready.set()
                # Wait for Stop, OR for the user to close the last page/window.
                while not stop_event.is_set():
                    if not context.pages:
                        break
                    time.sleep(0.3)
                try:
                    state = context.storage_state()
                    result_box["state"] = state
                except Exception as e:
                    result_box["err"] = str(e)
                try:
                    browser.close()
                except Exception:
                    pass
        except Exception as e:
            result_box["err"] = str(e)
            ready.set()

    t = threading.Thread(target=rec_thread, daemon=True)
    t.start()
    ready.wait(timeout=30)
    if result_box.get("err"):
        profile_rec_state = None
        return jsonify({"error": "Could not open browser: " + result_box["err"]}), 500
    profile_rec_state["thread"] = t
    profile_rec_state["result_box"] = result_box
    return jsonify({"message": "Browser opened — sign in, then click ‘Stop & save’ (or close the window)."})


@app.get("/api/profile/record/status")
def api_profile_record_status():
    if not profile_rec_state:
        return jsonify({"recording": False})
    try:
        current = profile_rec_state["page"].url
    except Exception:
        current = None
    return jsonify({"recording": True, "name": profile_rec_state["name"], "current": current})


@app.post("/api/profile/record/stop")
def api_profile_record_stop():
    global profile_rec_state
    if not profile_rec_state:
        return jsonify({"error": "No profile recording in progress."}), 400
    state = profile_rec_state
    name = state["name"]
    state["stop"].set()
    state["thread"].join(timeout=15)
    result_box = state["result_box"]
    profile_rec_state = None
    if result_box.get("err") or "state" not in result_box:
        return jsonify({"error": "Could not capture session: " + result_box.get("err", "no state")}), 500
    path, safe = _profile_path(name)
    write_json(path, result_box["state"])
    return jsonify({"message": f"Profile ‘{safe}’ saved", "name": safe, "profiles": list_profiles()})


# --- RECORD AS ANOTHER WINDOWS USER (automated, Windows-only) ----------------
# Best-effort: spawns a Chrome in another Windows user's session via `runas`.
# This is OS-specific and frequently blocked by policy; if it can't run we say
# so honestly rather than failing silently.
@app.post("/api/profile/runas")
def api_profile_runas():
    body = request.json or {}
    name, url = body.get("name"), body.get("url")
    win_user, win_pass = body.get("winUser"), body.get("winPass")
    if not all([name, url, win_user, win_pass]):
        return jsonify({"error": "Profile name, login URL, Windows user and password are all required"}), 400
    if sys.platform != "win32":
        return jsonify({"error": "RunAs is only available on Windows."}), 400
    # Honest stub: launching Chrome in *another* Windows user's interactive
    # session from a service process requires elevated rights + the interactive
    # desktop, which a dev server typically can't obtain. We surface that clearly
    # instead of pretending it worked. (Wire to a real RunAs helper when ready.)
    return jsonify({
        "error": "RunAs-as-another-user automation isn't enabled on this server yet. "
                 "Use ‘Record a new profile’ in a headed browser instead, or run PulseLab "
                 "inside that Windows user's own session.",
        "stub": True,
    }), 501


# --- On-demand per-page AI insights -----------------------------------------
# Reports render data-only; each page card has a "✨ Get AI insights" button that
# POSTs that page's audit digest here. We run the LLM and return the structured
# analysis, so no tokens are spent until a reader actually asks for a page.
@app.post("/api/page-insights")
def api_page_insights():
    body = request.json or {}
    digest = body.get("digest")
    if not isinstance(digest, dict):
        return jsonify({"error": "Missing page data."}), 400
    if not llm_enabled():
        return jsonify({"error": "AI insights need an Azure OpenAI key in pulselab/.env. "
                                 "Add one and restart PulseLab, then try again."}), 503
    analysis = _llm_analysis_from_digest(digest)
    if not analysis:
        return jsonify({"error": "The LLM couldn't produce insights for this page. Please try again."}), 502
    return jsonify({"analysis": analysis})


# --- Audit one URL: Lighthouse (lab) + CDP (live) ---------------------------
@app.post("/api/audit")
def api_audit():
    global audit_running, _WANT_AI
    body = request.json or {}
    url = body.get("url")
    device = body.get("device", "desktop")
    throttling = body.get("throttling", False)
    prefer = body.get("browser", "auto")
    headed = body.get("headed", False)
    _WANT_AI = bool(body.get("ai", False))  # AI insights opt-in
    if not url:
        return jsonify({"error": "Missing url"}), 400
    if audit_running:
        return jsonify({"error": "Another audit is already running. Please wait."}), 409
    audit_running = True

    try:
        with sync_playwright() as pw:
            # One Chrome with a fixed debug port; Lighthouse attaches to the same one.
            port = _free_port()
            browser = launch_browser(pw, [f"--remote-debugging-port={port}"], prefer, headed)
            ctx = browser.new_context(viewport={"width": 1350, "height": 940})
            page = ctx.new_page()
            thresholds = read_thresholds()

            # CDP pass (best-effort).
            cdp = None
            try:
                cdp = capture_cdp(page, url, thresholds)
            except Exception:
                pass

            # Lighthouse pass on the same Chrome.
            html, json_str, lhr = run_lighthouse(url, device, throttling, port)
            report_url = save_report(url, html)
            json_url = save_file(f"lhr-{url}", "json", json_str) if json_str else None

            page_data = summarise_page("Entry", url, lhr, thresholds, report_url, json_url, cdp)

            safe_name = re.sub(r"[^a-z0-9]", "_", re.sub(r"^https?://", "", url), flags=re.I)[:40] or "audit"
            overall_html = build_journey_report(url, [page_data], device)
            html_url = save_report(f"audit-{safe_name}", overall_html)
            md_url = save_file(f"audit-{safe_name}", "md", build_journey_markdown(url, [page_data], device))
            jsn_url = save_file(f"audit-{safe_name}", "json", json_lib.dumps(
                {"url": url, "device": device, "date": now_iso(), "pages": [page_data]}, indent=2))
            pdf_url = save_pdf(pw, f"audit-{safe_name}", overall_html)

            browser.close()

        run = {
            "kind": "audit", "name": url, "date": now_iso(), "pages": 1,
            "score": page_data["performanceScore"],
            "avgLcp": metric_numeric(page_data, "lcp"), "avgCls": metric_numeric(page_data, "cls"),
            "reportUrl": html_url,
            "formats": {"html": html_url, "pdf": pdf_url, "md": md_url, "json": jsn_url},
        }
        record_run(run)
        audit_running = False
        return jsonify({"message": "Audit complete", "mode": "single", "run": run,
                        "pages": [page_data], "thresholds": thresholds, "browserUsed": LAST_BROWSER})
    except Exception as err:
        audit_running = False
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(err)}), 500


# --- Audit a saved flow: Playwright traversal + per-page Lighthouse + CDP ----
@app.post("/api/traverse")
def api_traverse():
    global audit_running, _WANT_AI
    body = request.json or {}
    flow_file = body.get("flow")
    device = body.get("device", "desktop")
    prefer = body.get("browser", "auto")
    headed = body.get("headed", False)
    _WANT_AI = bool(body.get("ai", False))  # AI insights opt-in
    if not flow_file:
        return jsonify({"error": "Missing flow"}), 400
    if audit_running:
        return jsonify({"error": "Another audit is already running. Please wait."}), 409

    flow_path = os.path.join(FLOWS_DIR, os.path.basename(flow_file))
    if not os.path.exists(flow_path):
        return jsonify({"error": f"Flow not found: {flow_file}"}), 404
    audit_running = True
    live_reset(os.path.basename(flow_file))

    pages = []
    try:
        with sync_playwright() as pw:
            thresholds = read_thresholds()
            port = _free_port()
            browser = launch_browser(pw, [f"--remote-debugging-port={port}"], prefer, headed)
            context = browser.new_context(viewport={"width": 1350, "height": 940})
            page = context.new_page()

            ext = os.path.splitext(flow_path)[1].lower()
            setup_fn = None
            if ext == ".py":
                # Import the Python flow, RUN it once (establishes any login
                # session on this Chrome), then read its AUDIT_POINTS.
                mod = load_py_flow(flow_path)
                try:
                    mod.run(page, context, lambda m: live_step(m, page.url, None))
                except Exception as run_err:
                    # A flow step failing shouldn't abort the audit — we still
                    # cold-load each AUDIT_POINT below with whatever session exists.
                    print(f"[flow] run() raised (continuing to audit points): {run_err}")
                audit_points = getattr(mod, "AUDIT_POINTS", None) or [{"name": "start", "url": page.url}]
                setup_fn = getattr(mod, "setup", None) if callable(getattr(mod, "setup", None)) else None
            else:
                audit_points = resolve_audit_points_json(flow_path)

            if not audit_points or not audit_points[0].get("url"):
                raise RuntimeError("Flow has no auditable URLs")

            for ap in audit_points:
                cdp = None  # captured before Lighthouse; used as the fallback below
                try:
                    # Optional: re-establish the login session before scoring a
                    # protected page (so it loads for real instead of redirecting).
                    if setup_fn:
                        try:
                            setup_fn(page, context)
                        except Exception:
                            pass
                    cdp = capture_cdp(page, ap["url"], thresholds)
                    live_step(ap["name"], ap["url"], cdp.get("screenshot"))

                    # Score with the NODE API first: it drives the SAME
                    # authenticated Chrome (page is already on ap["url"] from
                    # capture_cdp above), so login-gated pages load for real
                    # instead of bouncing to /login. Fall back to the CLI (with a
                    # session cookie header) only if the Node runner is
                    # unavailable / errors.
                    html = json_str = lhr = None
                    lh_engine = None
                    try:
                        html, json_str, lhr = run_lighthouse_node(ap["url"], device, False, port)
                        lh_engine = "node"
                    except Exception as node_err:
                        print(f"[lh] node runner failed for {ap['name']} ({node_err}); trying CLI")
                        cookies = context.cookies()
                        cookie_header = "; ".join(f'{c["name"]}={c["value"]}' for c in cookies)
                        html, json_str, lhr = run_lighthouse(ap["url"], device, False, port, cookie_header)
                        lh_engine = "cli"

                    # A Lighthouse runtimeError (e.g. site rate-limiting with a
                    # 403 -> ERRORED_DOCUMENT_REQUEST) means the page never loaded
                    # reliably. Don't record the misleading score-0 report — fall
                    # back to the live CDP capture instead.
                    rt_err = (lhr.get("runtimeError") or {}).get("code")
                    perf_score = ((lhr.get("categories") or {}).get("performance") or {}).get("score")
                    lh_failed = bool(rt_err) or perf_score is None

                    lh_final = (lhr.get("finalDisplayedUrl") or lhr.get("finalUrl") or "")
                    redirected = (
                        bool(ap["url"]) and lh_final.rstrip("/") != ap["url"].rstrip("/")
                        and re.search(r"/(\?|$)", lh_final) and ap["url"] != lh_final
                        and ap["name"].lower() != "login"
                    )
                    if redirected or lh_failed:
                        # Still bounced (e.g. CLI fallback on a gated page) → keep
                        # the honest live CDP measurement.
                        pages.append(summarise_page(ap["name"], ap["url"], lhr, thresholds,
                                                    cdp=cdp, lighthouse_skipped=True))
                    else:
                        report_url = save_report(ap["name"], html)
                        json_url = save_file(f"lhr-{ap['name']}", "json", json_str) if json_str else None
                        pages.append(summarise_page(ap["name"], ap["url"], lhr, thresholds,
                                                    report_url, json_url, cdp))
                except Exception as err:
                    # If Lighthouse couldn't score the page (e.g. a login-gated /
                    # state-dependent page like checkout-complete that can't be
                    # cold-loaded — "Target closed" / navigation errors), fall back
                    # to the live CDP capture instead of a hard red error, so the
                    # page shows "CDP ✓ live" like the other protected pages.
                    if cdp and cdp.get("metrics"):
                        pages.append(summarise_page(ap["name"], ap["url"],
                                                    {"categories": {}, "audits": {}}, thresholds,
                                                    cdp=cdp, lighthouse_skipped=True))
                    else:
                        pages.append({"name": ap["name"], "url": ap["url"], "error": str(err),
                                      "metrics": [], "categories": [], "opportunities": [], "insights": []})

            ok = [p for p in pages if not p.get("error")]

            def avg(sel):
                if not ok:
                    return None
                return round(sum(sel(p) or 0 for p in ok) / len(ok))

            flow_base = os.path.splitext(os.path.basename(flow_file))[0]
            overall_html = build_journey_report(os.path.basename(flow_file), pages, device)
            journey_url = save_report(f"journey-{flow_base}", overall_html)
            journey_md = save_file(f"journey-{flow_base}", "md",
                                   build_journey_markdown(os.path.basename(flow_file), pages, device))
            journey_json = save_file(f"journey-{flow_base}", "json", json_lib.dumps(
                {"flow": os.path.basename(flow_file), "device": device, "date": now_iso(), "pages": pages}, indent=2))
            journey_pdf = save_pdf(pw, f"journey-{flow_base}", overall_html)

            browser.close()

        per_page = [{"name": p["name"], "url": p["reportUrl"]} for p in ok if p.get("reportUrl")]
        reports = [{"name": "🧭 Overall journey", "url": journey_url}] + per_page
        run = {
            "kind": "flow", "name": os.path.basename(flow_file), "date": now_iso(), "pages": len(pages),
            "score": avg(lambda p: p.get("performanceScore")),
            "avgLcp": avg(lambda p: metric_numeric(p, "lcp")),
            "avgCls": (sum(metric_numeric(p, "cls") or 0 for p in ok) / len(ok)) if ok else None,
            "reportUrl": journey_url,
            "reports": reports,
            "formats": {"html": journey_url, "pdf": journey_pdf, "md": journey_md, "json": journey_json},
        }
        record_run(run)
        audit_running = False
        LIVE["running"] = False
        return jsonify({"message": "Traversal complete", "mode": "flow", "run": run,
                        "pages": pages, "thresholds": thresholds, "browserUsed": LAST_BROWSER})
    except Exception as err:
        audit_running = False
        LIVE["running"] = False
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(err)}), 500


def _free_port():
    """Grab a free localhost port to pin Chrome's remote-debugging port to, so
    the Lighthouse CLI can attach to the exact Chrome we control."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# Saved HTML reports (index + static files).
@app.get("/reports")
def reports_index():
    files = []
    if os.path.exists(REPORTS_DIR):
        files = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith(".html")], reverse=True)
    items = "".join(f'<li><a href="/reports/{f}" target="_blank">{f}</a></li>' for f in files)
    return (f'<!doctype html><meta charset="utf-8"><title>Reports</title>'
            f"<h1>Saved Lighthouse reports</h1><ul>{items}</ul>"
            f'<p><a href="/">Back to PulseLab</a></p>')


@app.get("/reports/<path:filename>")
def reports_static(filename):
    return send_from_directory(REPORTS_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"PulseLab (Python) listening: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)
