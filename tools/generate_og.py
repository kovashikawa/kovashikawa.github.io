#!/usr/bin/env python3
"""generate_og — per-post Open Graph card generator for kovashikawa.github.io.

Renders a 1200x630 social preview card from a post's front matter, in the
blog's visual language: graph-paper grid over #E8E8E8, Inter 900 title,
JetBrains Mono signature, one steel-blue accent rule.

Pipeline (crisp by construction):
  1. Build HTML (blog palette + fonts) sized 1200x630.
  2. Screenshot in headless Chrome at 3x (--force-device-scale-factor=3)
     -> 3600x1890 master.
  3. LANCZOS-downscale to exactly 1200x630, save JPEG q=85 (<300 KB).

Usage:
  python3 generate_og.py _posts/2026-08-19-mcp-grew-up-fast.md
  python3 generate_og.py _posts/2026-08-19-mcp-grew-up-fast.md --out custom.jpg

Output: assets/images/og/<slug>.jpg where slug is the post filename minus
the date prefix and extension.

Dependencies: Python stdlib + Pillow + Google Chrome. Self-bootstraps a
venv next to this script on first run (installs Pillow).
"""

import argparse
import datetime
import pathlib
import re
import subprocess
import sys

BASE = pathlib.Path(__file__).resolve().parent
REPO = BASE.parent
OUT_DIR = REPO / "assets" / "images" / "og"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

W, H = 1200, 630
SCALE = 3

# Blog palette (custom.scss)
BG = "#E8E8E8"
GRID = "#ccc"
INK = "#1E1E1E"
MUTED = "#555"
ACCENT = "#2a78d6"


def ensure_pillow():
    """Self-bootstrap: venv next to this script with Pillow installed."""
    try:
        import PIL  # noqa: F401
        return [sys.executable]
    except ImportError:
        pass
    venv = BASE / ".venv"
    py = venv / "bin" / "python"
    if not py.exists():
        sys.stderr.write("generate_og: installing Pillow into tools/.venv (first run only)\n")
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        pip = venv / "bin" / "pip"
        subprocess.run([str(pip), "install", "--quiet", "Pillow"], check=True)
    return [str(py)]


def parse_front_matter(path):
    """Minimal YAML front matter reader: title, date, excerpt."""
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not m:
        raise SystemExit(f"no front matter in {path}")
    fm = m.group(1)

    def grab(key):
        mm = re.search(rf'^{key}:\s*["\']?(.+?)["\']?\s*$', fm, re.MULTILINE)
        return mm.group(1).strip() if mm else None

    title = grab("title") or path.stem
    date_raw = grab("date") or ""
    dmatch = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_raw)
    date_str = ""
    if dmatch:
        y, mo, d = map(int, dmatch.groups())
        date_str = datetime.date(y, mo, d).strftime("%B %-d, %Y")
    return title, date_str


def title_font_size(title):
    n = len(title)
    if n <= 38:
        return 76
    if n <= 64:
        return 64
    if n <= 100:
        return 54
    if n <= 140:
        return 46
    return 40


def build_html(title, date_str):
    fs = title_font_size(title)
    date_html = f'<div class="date">{date_str}</div>' if date_str else ""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@600;900&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {W}px; height: {H}px; overflow: hidden; }}
  body {{
    background-color: {BG};
    background-image:
      linear-gradient({GRID} 1px, transparent 1px),
      linear-gradient(90deg, {GRID} 1px, transparent 1px);
    background-size: 50px 50px;
    font-family: 'Inter', sans-serif;
    position: relative;
  }}
  .card {{
    position: absolute;
    inset: 0;
    background: {BG};
    /* solid plate keeps the title legible over the grid */
    opacity: 0;
  }}
  .content {{
    position: absolute;
    left: 84px;
    right: 84px;
    top: 0;
    bottom: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }}
  .domain {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 21px;
    color: {MUTED};
    letter-spacing: 0.04em;
    margin-bottom: 34px;
  }}
  .title {{
    font-weight: 900;
    font-size: {fs}px;
    line-height: 1.14;
    color: {INK};
    letter-spacing: -0.02em;
    max-width: 100%;
  }}
  .rule {{
    width: 132px;
    height: 7px;
    background: {ACCENT};
    margin-top: 36px;
  }}
  .date {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 400;
    font-size: 19px;
    color: {MUTED};
    margin-top: 22px;
  }}
</style>
</head>
<body>
  <div class="content">
    <div class="domain">kovashikawa.github.io</div>
    <div class="title">{title}</div>
    <div class="rule"></div>
    {date_html}
  </div>
</body>
</html>"""


def slugify(post_path):
    stem = pathlib.Path(post_path).stem          # 2026-08-19-mcp-grew-up-fast
    return re.sub(r"^\d{4}-\d{2}-\d{2}-", "", stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("post", help="path to the post markdown file")
    ap.add_argument("--out", help="output jpg path (default assets/images/og/<slug>.jpg)")
    args = ap.parse_args()

    post = pathlib.Path(args.post)
    title, date_str = parse_front_matter(post)
    out = pathlib.Path(args.out) if args.out else OUT_DIR / f"{slugify(post)}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)

    html_path = BASE / "_og_tmp.html"
    html_path.write_text(build_html(title, date_str), encoding="utf-8")

    master = BASE / "_og_master.png"
    if master.exists():
        master.unlink()
    url = f"file://{html_path}?v={datetime.datetime.now().timestamp()}"

    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--force-device-scale-factor={SCALE}",
            f"--window-size={W},{H}",
            "--virtual-time-budget=15000",
            f"--screenshot={master}",
            url,
        ],
        check=True,
        capture_output=True,
    )

    py = ensure_pillow()
    code = (
        "from PIL import Image\n"
        f"im = Image.open(r'{master}')\n"
        f"im = im.resize(({W}, {H}), Image.Resampling.LANCZOS)\n"
        f"im.convert('RGB').save(r'{out}', 'JPEG', quality=85, optimize=True)\n"
    )
    subprocess.run(py + ["-c", code], check=True)

    kb = out.stat().st_size // 1024
    print(f"wrote {out} ({kb} KB)")
    html_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
