#!/usr/bin/env python3
"""Build the public research log for https://learning.mywords.io from this repo's docs.

Usage (from the repo root, with the venv that has `markdown`):
    .venv/bin/python tools/publish_site.py [--out /home/ezio/apps/learning] [--commit SHA]

Pure static output, served as-is by a static file server behind Cloudflare.
Rewriting the files is the whole deploy: no restart. The swap is atomic (build dir + rename).
Public-content rules: byline "Ez" only, times as UTC+8 (never HKT / Hong Kong), no infra details.
"""
import argparse, datetime, html, os, re, shutil, subprocess, sys, tempfile
import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_URL = "https://github.com/CLQuantizer/pdp-continual"
TZ = datetime.timezone(datetime.timedelta(hours=8))

DOCS = [  # (slug, source, title, blurb)
    ("loop", "docs/loop.md", "Hourly loop log", "What each hour learned, killed, and queued next."),
    ("goal", "docs/goal.md", "Goal", "A machine that lives, not a model that is trained."),
    ("notes", "docs/notes.md", "Research notes", "Longer-form findings and results."),
    ("audit", "audit.md", "Audit", "Claims checked against evidence; open items."),
    ("cls-route", "docs/cls_route.md", "CLS routing", "Answering with the hippocampal store."),
    ("readme", "README.md", "Full write-up", "The long README: methods, runs, numbers."),
    ("architecture", "docs/architecture.md", "Architecture", "How the research runs: Claude Code orchestrates grok and codex, Kaggle computes, Cloudflare serves and stores, Telegram reports."),
    ("how-it-works", "docs/how_it_works.md", "How it works", "One-glance figure of the dual: store, router, uniform awake replay."),
    ("h20", "docs/h20_world.md", "H20: a new world", "Pre-registered next: a label-free algorithmic stream, scored in bits per symbol."),
    ("roadmap", "docs/roadmap.md", "Roadmap", "From parts to one curious machine: milestones M1-M5 and their kill conditions."),
    ("literature", "docs/literature_2026-11.md", "Literature scan, Nov 2026", "Progress estimators, fast weights, e-prop, Titans, loss of plasticity: what each changes."),
    ("curiosity", "docs/curiosity.md", "Guideline: compression progress", "Schmidhuber 2009 as working rules: store, compress, reward progress, act on it."),
    ("h23", "docs/h23_curiosity.md", "H23: curiosity", "An active world with a noisy TV: progress-seeking vs error-seeking attention."),
    ("hippocampus", "docs/hippocampus_diagnosis.md", "Why the hippocampus looked useless", "Diagnosis: no episodes in the data, an averaging store, a rival-answerer test."),
]

SCRUB = [
    (re.compile(r"\bHKT\s*\(UTC\+8\)"), "UTC+8"),
    (re.compile(r"\bHKT\b"), "UTC+8"),
    (re.compile(r"\bHong\s*Kong\b", re.I), ""),
    (re.compile(r"\bEzQ\b"), "Ez"),
    (re.compile(r"\b[Dd]ubai\s+(VPS|box|server)\b"), "Linux server"),
    (re.compile(r"\b(on|from|redone on) dubai\b"), r"\1 the Linux server"),
    (re.compile(r"\bdubai runs\b"), "Linux-server runs"),
    (re.compile(r"\bdubai\b", re.I), "Linux server"),
    (re.compile(r"\b10\.\d+\.\d+\.\d+\b"), "[server]"),
    (re.compile(r"/root/pdp/?"), ""),
    (re.compile(r"/home/\w+/"), "~/"),
    (re.compile(r"[\w.+-]+@(gmail|outlook|hotmail)\.com", re.I), "[email]"),
]

def scrub(text):
    for pat, rep in SCRUB:
        text = pat.sub(rep, text)
    return text

CSS = """
:root{--fg:#1d1d1f;--mut:#6e6e73;--bg:#fbfbf8;--acc:#0b5cad;--line:#e4e2dc;--code:#f2f0ea}
@media (prefers-color-scheme:dark){:root{--fg:#e8e6e1;--mut:#9a978f;--bg:#151514;--acc:#7fb4ff;--line:#2c2b29;--code:#201f1d}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.6 Charter,"Iowan Old Style",Georgia,serif}
.wrap{max-width:780px;margin:0 auto;padding:28px 20px 80px}
nav{font:14px/1.4 -apple-system,system-ui,sans-serif;display:flex;flex-wrap:wrap;gap:14px;border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:28px}
nav a{color:var(--mut);text-decoration:none}nav a:hover,nav a.on{color:var(--fg)}
nav .brand{color:var(--fg);font-weight:600;margin-right:auto}
a{color:var(--acc)}h1,h2,h3,h4{font-family:-apple-system,system-ui,sans-serif;line-height:1.25;margin:1.8em 0 .5em}
h1{font-size:1.9em;margin-top:.2em}h2{font-size:1.35em;border-top:1px solid var(--line);padding-top:1.1em}h3{font-size:1.1em}
code{font:0.86em/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--code);padding:.1em .3em;border-radius:4px}
pre{background:var(--code);padding:12px 14px;border-radius:6px;overflow-x:auto}pre code{padding:0;background:none}
table{border-collapse:collapse;display:block;overflow-x:auto;font:14px/1.4 -apple-system,system-ui,sans-serif;margin:1em 0}
th,td{border:1px solid var(--line);padding:5px 9px;text-align:left;vertical-align:top}th{background:var(--code)}
blockquote{margin:1em 0;padding:.1em 1em;border-left:3px solid var(--line);color:var(--mut)}
.meta,footer{font:13px/1.5 -apple-system,system-ui,sans-serif;color:var(--mut)}
footer{border-top:1px solid var(--line);margin-top:48px;padding-top:14px}
.cards{list-style:none;padding:0}.cards li{margin:0 0 14px}.cards a{font-weight:600}
.toc{font:14px/1.5 -apple-system,system-ui,sans-serif;columns:2;column-gap:28px;padding-left:1.1em}
.latest{border:1px solid var(--line);border-radius:8px;padding:2px 18px 10px;margin:22px 0}
.latest h2{border:0;padding-top:0}
img{max-width:100%}
iframe.fig{width:100%;height:640px;border:0;display:block;margin:0 auto;background:transparent}
.feature{margin:22px 0 8px}
"""

NAV = [("", "Home"), ("how-it-works/", "How it works"), ("architecture/", "Architecture"), ("loop/", "Loop log"), ("results/", "Results"), ("goal/", "Goal"), ("notes/", "Notes"), ("audit/", "Audit")]

def md(text):
    return markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "toc"],
                             extension_configs={"toc": {"permalink": False}})

def link_fix(h, depth):
    up = "../" * depth
    def rep(m):
        href = m.group(1)
        if re.match(r"^[a-z]+:|^#|^/", href):
            return m.group(0)
        mm = re.search(r"results/([\w.-]+)/RESULTS\.md", href)
        if mm:
            return f'href="{up}results/{mm.group(1)}/"'
        for slug, src, *_ in DOCS:
            if href.rstrip("/").endswith(os.path.basename(src)):
                return f'href="{up}{slug}/"'
        if href.endswith((".py", ".md")) or "/" in href:
            return f'href="{REPO_URL}/blob/main/{href.lstrip("./")}"'
        return m.group(0)
    return re.sub(r'href="([^"]+)"', rep, h)

def page(title, body, depth, on, stamp):
    up = "../" * depth
    nav = "".join(f'<a href="{up}{p}"{" class=on" if p == on else ""}>{html.escape(n)}</a>' for p, n in NAV[1:])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title) + " · " if title != "Learning" else ""}Learning, by Ez</title>
<meta name="description" content="A live research log: backprop-free continual learning with wake/sleep cycles.">
<meta name="author" content="Ez">
<style>{CSS}</style></head><body><div class="wrap">
<nav><a class="brand" href="{up or './'}">Learning <span style="font-weight:400;color:var(--mut)">by Ez</span></a>{nav}</nav>
<main>{body}</main>
<script>addEventListener("message",function(e){{var d=e.data;if(!d||d.type!=="cls-figure-height")return;document.querySelectorAll("iframe.fig").forEach(function(f){{if(f.contentWindow===e.source)f.style.height=(d.height+4)+"px"}})}})</script>
<footer>By Ez · Built from <a href="{REPO_URL}">pdp-continual</a>{stamp} · Updated as the research runs.</footer>
</div></body></html>"""

def write(out, rel, content):
    p = os.path.join(out, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)

def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return scrub(f.read())

def loop_entries(text):
    parts = re.split(r"(?m)^(?=## )", text)
    head, entries = parts[0], [p for p in parts[1:] if p.strip()]
    return head, entries

def first_line(text):
    for ln in text.splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and not ln.startswith("|") and not ln.startswith("---"):
            return re.sub(r"[*`_]", "", ln)[:220]
    return ""

def build(out, commit):
    now = datetime.datetime.now(TZ).strftime("%Y-%m-%d %H:%M UTC+8")
    stamp = f' @ <a href="{REPO_URL}/commit/{commit}">{commit[:7]}</a>' if commit else ""
    stamp += f" · {now}"

    # Loop log: newest first, with a table of contents.
    head, entries = loop_entries(read("docs/loop.md"))
    entries = entries[::-1]
    toc, chunks = [], []
    for i, e in enumerate(entries):
        title = e.splitlines()[0][3:].strip()
        anchor = f"e{len(entries) - i}"
        toc.append(f'<li><a href="#{anchor}">{html.escape(title)}</a></li>')
        chunks.append(f'<section id="{anchor}">' + link_fix(md(e), 1) + "</section>")
    loop_body = ("<h1>Hourly loop log</h1><p class=meta>Each hour: what was learned, what was killed, "
                 "what to test next. Newest first.</p>" + f'<ol class="toc" reversed>{"".join(toc)}</ol>' + "".join(chunks))
    write(out, "loop/index.html", page("Hourly loop log", loop_body, 1, "loop/", stamp))

    for slug, src, title, _ in DOCS:
        if slug == "loop" or not os.path.exists(os.path.join(ROOT, src)):
            continue
        write(out, f"{slug}/index.html", page(title, link_fix(md(read(src)), 1), 1, f"{slug}/", stamp))

    # Results: one page per results/*/RESULTS.md, index sorted by last change.
    rdir = os.path.join(ROOT, "results")
    rows = []
    for name in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
        f = os.path.join(rdir, name, "RESULTS.md")
        if not os.path.isfile(f):
            continue
        text = read(f"results/{name}/RESULTS.md")
        mtime = os.path.getmtime(f)
        rows.append((mtime, name, first_line(text)))
        body = f'<p class=meta><a href="../">← all results</a> · <code>results/{html.escape(name)}</code></p>' + link_fix(md(text), 2)
        write(out, f"results/{name}/index.html", page(f"Results: {name}", body, 2, "results/", stamp))
    rows.sort(reverse=True)
    lis = "".join(
        f'<li><a href="{n}/">{html.escape(n)}</a> <span class=meta>{datetime.datetime.fromtimestamp(t, TZ):%Y-%m-%d %H:%M}</span><br>{html.escape(d)}</li>'
        for t, n, d in rows)
    write(out, "results/index.html", page("Results", f"<h1>Results</h1><p class=meta>{len(rows)} experiment write-ups, most recently updated first. Times UTC+8.</p><ul class=cards>{lis}</ul>", 1, "results/", stamp))

    # Home: intro + latest loop entry + map.
    goal = read("docs/goal.md")
    north = re.search(r"(?ms)^## North star.*?(?=^## )", goal)
    latest = link_fix(md(entries[0]), 0) if entries else ""
    cards = "".join(f'<li><a href="{s}/">{html.escape(t)}</a><br><span class=meta>{html.escape(b)}</span></li>'
                    for s, src, t, b in DOCS if os.path.exists(os.path.join(ROOT, src)))
    cards += f'<li><a href="results/">Results</a><br><span class=meta>{len(rows)} experiment write-ups.</span></li>'
    home = ("<h1>Learning</h1><p>A live research log by Ez on <b>backprop-free continual learning with wake/sleep "
            "cycles</b>: a network that answers and learns at once, keeps a hippocampus-like fast store, and sleeps "
            "when it needs to. An hourly loop runs experiments, pre-registers kill tests, and writes down what died. "
            f'This site republishes as the loop commits. Code: <a href="{REPO_URL}">pdp-continual</a>.</p>'
            + ('<div class="feature"><iframe class="fig" src="figures/cls-bench.html" title="Fig 1: CLS dual bench, store, router, uniform awake replay" loading="lazy"></iframe>'
               '<p class=meta>Fig 1 · how the default core works at one glance · <a href="how-it-works/">read the figure</a> · '
               '<a href="figures/cls-bench.html">open full size</a></p></div>' if os.path.exists(os.path.join(ROOT, "docs/figures/cls-bench.html")) else "")
            + (link_fix(md(north.group(0)), 0) if north else "")
            + f'<div class="latest"><p class=meta>Latest loop entry · <a href="loop/">full log ({len(entries)} entries)</a></p>{latest}</div>'
            + f"<h2>Read</h2><ul class=cards>{cards}</ul>")
    write(out, "index.html", page("Learning", home, 0, "", stamp))
    write(out, "robots.txt", "User-agent: *\nAllow: /\n")
    fdir = os.path.join(ROOT, "docs", "figures")
    for name in sorted(os.listdir(fdir)) if os.path.isdir(fdir) else []:
        if name.endswith(".html"):
            write(out, f"figures/{name}", read(f"docs/figures/{name}"))
    return len(entries), len(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ezio/apps/learning")
    ap.add_argument("--commit", default="")
    ap.add_argument("--owner", default="ezio")
    a = ap.parse_args()
    commit = a.commit
    if not commit:
        try:
            commit = subprocess.check_output(["git", "-c", f"safe.directory={ROOT}", "-C", ROOT, "rev-parse", "HEAD"],
                                             text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            commit = ""
    parent = os.path.dirname(os.path.abspath(a.out))
    tmp = tempfile.mkdtemp(prefix=".learning-build-", dir=parent)
    n_loop, n_res = build(tmp, commit)
    if a.owner and os.geteuid() == 0:
        subprocess.run(["chown", "-R", f"{a.owner}:{a.owner}", tmp], check=True)
    os.chmod(tmp, 0o755)
    old = a.out + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(a.out):
        os.rename(a.out, old)
    os.rename(tmp, a.out)
    shutil.rmtree(old, ignore_errors=True)
    print(f"published {n_loop} loop entries, {n_res} results -> {a.out} ({commit[:7] or 'no commit'})")

if __name__ == "__main__":
    sys.exit(main())
