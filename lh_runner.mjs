/**
 * Lighthouse Node-API runner — scores LOGIN-GATED pages that the CLI can't.
 *
 * The Lighthouse CLI audits in an ISOLATED browser context that does not inherit
 * the authenticated session's cookies, so SauceDemo (and any cookie-gated app)
 * bounces it to /login. The Node API lets us hand Lighthouse a puppeteer `page`
 * that lives in the SAME context our Playwright flow logged into — so the audit
 * runs with the real session and the gated page loads for real.
 *
 * Usage (called by server.py):
 *   node lh_runner.mjs --port=<cdpPort> --url=<auditUrl> --device=desktop \
 *                      --out=<basePath> [--no-throttle]
 * Writes <basePath>.report.html and <basePath>.report.json, prints one JSON
 * line to stdout: {"ok":true,"json":"<path>","html":"<path>","finalUrl":"..."}.
 *
 * Lighthouse 11 is ESM and bundles puppeteer-core + chrome-launcher, so we
 * import everything from the global lighthouse install (resolved via its path,
 * passed as --lhdir so server.py controls which install we use).
 */
import { writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

function arg(name, def = undefined) {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  if (hit) return hit.slice(name.length + 3);
  return process.argv.includes(`--${name}`) ? true : def;
}

const port = Number(arg('port'));
const url = arg('url');
const device = arg('device', 'desktop');
const outBase = arg('out');
const noThrottle = arg('no-throttle', false);
const lhDir = arg('lhdir'); // absolute path to the lighthouse module dir

if (!port || !url || !outBase || !lhDir) {
  console.log(JSON.stringify({ ok: false, error: 'missing --port/--url/--out/--lhdir' }));
  process.exit(2);
}

async function main() {
  // Import lighthouse + its bundled puppeteer-core from the resolved install.
  const lighthouse = (await import(pathToFileURL(path.join(lhDir, 'core', 'index.js')).href)).default;
  const puppeteer = (await import(
    pathToFileURL(path.join(lhDir, 'node_modules', 'puppeteer-core', 'lib', 'esm', 'puppeteer', 'puppeteer-core.js')).href
  )).default;

  // Connect to the SAME Chrome our Playwright flow authenticated.
  const browser = await puppeteer.connect({
    browserURL: `http://127.0.0.1:${port}`,
    defaultViewport: null,
  });

  // Reuse an existing authenticated tab if one is open; else open one in the
  // default (authenticated) context so cookies are shared.
  const pages = await browser.pages();
  const page = pages.find((p) => /saucedemo|^https?:/i.test(p.url())) || (await browser.newPage());

  const screenEmulation = device === 'mobile'
    ? { mobile: true, width: 412, height: 823, deviceScaleFactor: 1.75, disabled: false }
    : { mobile: false, width: 1350, height: 940, deviceScaleFactor: 1, disabled: false };

  const flags = {
    output: ['html', 'json'],
    logLevel: 'error',
    onlyCategories: ['performance', 'accessibility', 'best-practices', 'seo'],
    formFactor: device === 'mobile' ? 'mobile' : 'desktop',
    screenEmulation,
    disableStorageReset: true, // keep the session cookie — do NOT clear it
    // Lighthouse 11.7.1's RootCauses / TraceElements gatherer crashes on the
    // trace format emitted by the very new HeadlessChrome 148 ("Cannot read
    // properties of undefined (reading 'frame_sequence')"). That single gatherer
    // crash cascades into every audit that depends on it, each showing a
    // spurious red "Error!" row. ALL of these audits carry 0 weight in the
    // performance score, so skipping them changes NO numbers (score stays 100) —
    // it only removes the misleading error rows. (Pure version-mismatch between
    // LH11 and Chrome 148; not a problem with the page or our config.)
    skipAudits: [
      'prioritize-lcp-image',          // Preload Largest Contentful Paint image
      'largest-contentful-paint-element',
      'lcp-lazy-loaded',               // LCP image was not lazily loaded
      'layout-shift-elements',         // Avoid large layout shifts
      'layout-shifts',
      'non-composited-animations',     // Avoid non-composited animations
    ],
  };
  if (noThrottle) {
    // "No throttling" = a fast, FINITE connection applied via DevTools.
    //
    // Do NOT use throttlingMethod:'provided' here: under 'provided', the
    // byte-efficiency audits (Minify CSS/JS, text-compression, HTTP/2,
    // responsive/modern images) derive RTT from observed request timing rather
    // than from these `throttling` values. On a near-instant load that observed
    // RTT computes as Infinity, so every one of those audits errors with
    // "Invalid rtt Infinity". throttlingMethod:'devtools' actually applies the
    // finite rtt/throughput below over CDP, so the audits get a real RTT and
    // compute instead of erroring — while rttMs:1 + 100Mbps keeps it
    // effectively unthrottled.
    flags.throttlingMethod = 'devtools';
    flags.throttling = {
      rttMs: 1, throughputKbps: 100000, cpuSlowdownMultiplier: 1,
      requestLatencyMs: 1, downloadThroughputKbps: 100000, uploadThroughputKbps: 100000,
    };
  }

  // The 4th arg (page) is the key: Lighthouse audits THIS authenticated page's
  // context instead of spinning up an isolated one.
  //
  // Retry on TRANSIENT load failures (e.g. SauceDemo rate-limiting with HTTP
  // 403 -> ERRORED_DOCUMENT_REQUEST). A single 403 must not poison the run with
  // a misleading score of 0; retry a couple of times with backoff, and if it
  // still fails, throw so Python falls back to the live CDP capture.
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let runnerResult = null;
  let lastRuntimeErr = null;
  for (let attempt = 1; attempt <= 3; attempt++) {
    runnerResult = await lighthouse(url, flags, undefined, page);
    const rtErr = runnerResult && runnerResult.lhr && runnerResult.lhr.runtimeError;
    if (!rtErr) break;
    lastRuntimeErr = rtErr;
    if (attempt < 3) await sleep(4000 * attempt); // 4s, 8s backoff
  }
  if (!runnerResult || !runnerResult.report) throw new Error('Lighthouse returned no report');
  if (runnerResult.lhr && runnerResult.lhr.runtimeError) {
    const e = runnerResult.lhr.runtimeError;
    throw new Error(`runtimeError ${e.code}: ${e.message}`);
  }

  const [html, json] = runnerResult.report; // output order matches flags.output
  const jsonPath = `${outBase}.report.json`;
  const htmlPath = `${outBase}.report.html`;
  writeFileSync(jsonPath, json, 'utf-8');
  writeFileSync(htmlPath, html, 'utf-8');

  const lhr = runnerResult.lhr;
  console.log(JSON.stringify({
    ok: true,
    json: jsonPath,
    html: htmlPath,
    finalUrl: lhr.finalDisplayedUrl || lhr.finalUrl || '',
    score: (lhr.categories && lhr.categories.performance && lhr.categories.performance.score) ?? null,
  }));

  // Disconnect (do NOT close — it's the flow's browser).
  browser.disconnect();
}

main().catch((err) => {
  console.log(JSON.stringify({ ok: false, error: String(err && err.stack || err) }));
  process.exit(1);
});
