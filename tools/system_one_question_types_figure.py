# /// script
# dependencies = ["pillow"]
# ///
"""Figure for the Noul / Choice / Score post: one state, three question types, three answer shapes.

Run: uv run tools/system_one_question_types_figure.py
Renders SVG in HTML with headless Chrome at 3x and downscales with LANCZOS (html-to-image-rendering pipeline).
The probabilities are illustrative, as the figure says; the state and questions follow TypeSafe's docs example.
"""
import pathlib, subprocess, time
from PIL import Image

OUT = pathlib.Path(__file__).parent.parent / "assets" / "images" / "system-one-noul-choice-score"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
INK, MUTED, LIGHT, RULE, BASE = "#1E1E1E", "#555555", "#888888", "#e6e6e3", "#c9c9c4"
ACCENT, ACCENT_LT, ACCENT_BG = "#2a78d6", "#86b6ef", "#f0f6fd"
FONT = '-apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif'

W, H = 1600, 760
COL_W, GAP, X0 = 480, 40, 40
CARD_Y, CARD_H = 205, 440

STATE = "My flight was cancelled. Can I get a refund?"
columns = [
    dict(kind="NOUL", q=["Does the message request", "a refund?"], rows=[("yes", 0.92)],
         note="One number: the probability of yes.", accent=False, ordered=False),
    dict(kind="CHOICE", q=["What is the main request?", "Pick one option."],
         rows=[("refund", 0.80), ("rebooking", 0.15), ("information", 0.05)],
         note="One probability per option. No order.", accent=False, ordered=False),
    dict(kind="SCORE", q=["How frustrated does the", "customer appear?"],
         rows=[("0  calm", 0.30), ("1  concerned", 0.55), ("2  very angry", 0.15)],
         note="One probability per level, in order, plus a score of 0.85.", accent=True, ordered=True),
]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


def build():
    p = []
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family=\'{FONT}\'>')
    p.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    p.append('<defs>'
             f'<marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{LIGHT}"/></marker>'
             f'<marker id="ab" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{ACCENT}"/></marker>'
             '</defs>')
    # state box
    sx, sw, sy, sh = 40, 1520, 30, 74
    p.append(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" rx="12" fill="{INK}"/>')
    p.append(f'<text x="{sx+28}" y="{sy+45}" font-size="17" font-weight="700" letter-spacing="1.8" fill="{BASE}">THE STATE</text>')
    p.append(f'<text x="{sx+185}" y="{sy+46}" font-size="26" fill="#ffffff">&#8220;{esc(STATE)}&#8221;</text>')
    p.append(f'<text x="{sx+sw-28}" y="{sy+45}" font-size="17" text-anchor="end" fill="{BASE}">the content to judge</text>')
    # connectors and question labels
    p.append(f'<text x="330" y="152" font-size="18" fill="{MUTED}">The same state, three typed questions about it</text>')
    for i, c in enumerate(columns):
        cx = X0 + i * (COL_W + GAP)
        mid = cx + COL_W / 2
        col = ACCENT if c["accent"] else LIGHT
        mk = "ab" if c["accent"] else "ah"
        p.append(f'<path d="M{mid},{sy+sh+8} C{mid},{sy+sh+50} {mid},{CARD_Y-60} {mid},{CARD_Y-6}" stroke="{col}" stroke-width="2.2" fill="none" marker-end="url(#{mk})"/>')
        # card
        stroke = ACCENT_LT if c["accent"] else BASE
        fill = ACCENT_BG if c["accent"] else "#ffffff"
        p.append(f'<rect x="{cx}" y="{CARD_Y}" width="{COL_W}" height="{CARD_H}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        tcol = ACCENT if c["accent"] else INK
        p.append(f'<text x="{cx+28}" y="{CARD_Y+50}" font-size="26" font-weight="800" letter-spacing="2.4" fill="{tcol}">{c["kind"]}</text>')
        for k, line in enumerate(c["q"]):
            p.append(f'<text x="{cx+28}" y="{CARD_Y+88+k*26}" font-size="19" fill="{MUTED}">{esc(line)}</text>')
        # bars
        by = CARD_Y + 170
        label_w = 150
        track_x = cx + 28 + label_w
        track_w = COL_W - 28 - label_w - 28 - 62
        bar_h, step = 34, 58
        if c["ordered"]:
            # ordered axis arrow on the left edge
            ay0, ay1 = by - 10, by + step * (len(c["rows"]) - 1) + bar_h + 10
            p.append(f'<path d="M{cx+14},{ay0} L{cx+14},{ay1}" stroke="{ACCENT}" stroke-width="2.2" fill="none" marker-end="url(#ab)"/>')
        for r, (lab, v) in enumerate(c["rows"]):
            y = by + r * step
            p.append(f'<text x="{cx+28}" y="{y+24}" font-size="19" fill="{INK}">{esc(lab)}</text>')
            p.append(f'<rect x="{track_x}" y="{y}" width="{track_w}" height="{bar_h}" rx="5" fill="#f1f1ef"/>')
            bcol = ACCENT if c["accent"] else "#8a8a85"
            p.append(f'<rect x="{track_x}" y="{y}" width="{max(track_w*v,3):.1f}" height="{bar_h}" rx="5" fill="{bcol}"/>')
            p.append(f'<text x="{track_x+track_w+12}" y="{y+24}" font-size="19" font-weight="700" fill="{INK}">{v:.2f}</text>')
        # note
        note = c["note"]
        words = note.split()
        lines, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 44:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        ny = CARD_Y + CARD_H - 22 - (len(lines) - 1) * 24
        for k, ln in enumerate(lines):
            p.append(f'<text x="{cx+28}" y="{ny+k*24}" font-size="18" fill="{MUTED}">{esc(ln)}</text>')
    # footer
    p.append(f'<text x="40" y="{CARD_Y+CARD_H+52}" font-size="18" fill="{MUTED}">Illustrative probabilities. A System One model returns only probabilities, never text. Questions and state follow the example in TypeSafe\'s docs.</text>')
    p.append('</svg>')
    return "\n".join(p)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = str(int(time.time()))
    html = OUT / f"_fig_{stamp}.html"
    master = OUT / f"_master_{stamp}.png"
    html.write_text(f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:#fff}}</style></head><body>{build()}</body></html>')
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=3",
                    f"--window-size={W},{H}", "--virtual-time-budget=8000", f"--screenshot={master}", f"file://{html}?v={stamp}"],
                   check=True, capture_output=True)
    im = Image.open(master).convert("RGB").resize((W * 2, H * 2), Image.Resampling.LANCZOS)
    dest = OUT / "question-types.png"
    im.save(dest, optimize=True)
    html.unlink(); master.unlink()
    print("wrote", dest, im.size, dest.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
