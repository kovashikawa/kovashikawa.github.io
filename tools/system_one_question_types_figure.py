# /// script
# dependencies = ["pillow"]
# ///
"""Figure for the Noul / Choice / Score post: one state, three question types, three answer shapes.

Run: uv run tools/system_one_question_types_figure.py
Renders SVG in HTML with headless Chrome at 3x and downscales with LANCZOS (html-to-image-rendering pipeline).
The cards are stacked so the figure stays legible at phone width. The probabilities are illustrative;
the state and questions follow TypeSafe's docs example, which the post's caption says.
"""
import pathlib, subprocess, time
from PIL import Image

OUT = pathlib.Path(__file__).parent.parent / "assets" / "images" / "system-one-noul-choice-score"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
INK, MUTED, LIGHT, RULE, BASE = "#1E1E1E", "#555555", "#888888", "#e6e6e3", "#c9c9c4"
ACCENT, ACCENT_LT, ACCENT_BG = "#2a78d6", "#86b6ef", "#f0f6fd"
FONT = '-apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif'

W = 880
MARGIN, SPINE_X, CARD_X, CARD_R = 40, 58, 92, 840
PAD, GAP = 26, 18
BAR_H, STEP = 32, 50

STATE = "My flight was cancelled. Can I get a refund?"
cards = [
    dict(kind="NOUL", q="Does the message request a refund?", rows=[("yes", 0.92)],
         note="One number: the probability of yes.", accent=False, ordered=False),
    dict(kind="CHOICE", q="What is the main request? Pick one option.",
         rows=[("refund", 0.80), ("rebooking", 0.15), ("information", 0.05)],
         note="One probability per option. No order.", accent=False, ordered=False),
    dict(kind="SCORE", q="How frustrated does the customer appear?",
         rows=[("0  calm", 0.30), ("1  concerned", 0.55), ("2  very angry", 0.15)],
         note="One probability per level, in order, plus a score of 0.85.", accent=True, ordered=True),
]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


def card_height(c):
    rows_h = BAR_H + STEP * (len(c["rows"]) - 1)
    return 76 + rows_h + 56


def build():
    tops, y = [], 184
    for c in cards:
        tops.append(y)
        y += card_height(c) + GAP
    H = y - GAP + MARGIN

    p = []
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family=\'{FONT}\'>')
    p.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    p.append('<defs>'
             f'<marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{LIGHT}"/></marker>'
             f'<marker id="ab" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="{ACCENT}"/></marker>'
             '</defs>')
    # state box
    sx, sw, sy, sh = MARGIN, W - 2 * MARGIN, 30, 74
    p.append(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" rx="12" fill="{INK}"/>')
    p.append(f'<text x="{sx+26}" y="{sy+45}" font-size="16" font-weight="700" letter-spacing="1.8" fill="{BASE}">THE STATE</text>')
    p.append(f'<text x="{sx+158}" y="{sy+46}" font-size="26" fill="#ffffff">&#8220;{esc(STATE)}&#8221;</text>')
    # spine from the state down past the last card, one branch per card
    last_mid = tops[-1] + card_height(cards[-1]) / 2
    p.append(f'<path d="M{SPINE_X},{sy+sh+10} L{SPINE_X},{last_mid}" stroke="{LIGHT}" stroke-width="2.2" fill="none"/>')
    p.append(f'<text x="{CARD_X}" y="{sy+sh+54}" font-size="22" fill="{MUTED}">The same state, three typed questions about it</text>')
    inner_l, inner_r = CARD_X + PAD, CARD_R - PAD
    q_x, track_x, value_x = inner_l + 150, inner_l + 200, inner_r - 56
    track_w = value_x - 14 - track_x
    for c, top in zip(cards, tops):
        h = card_height(c)
        mid = top + h / 2
        col, mk = (ACCENT, "ab") if c["accent"] else (LIGHT, "ah")
        p.append(f'<path d="M{SPINE_X},{mid} L{CARD_X-4},{mid}" stroke="{col}" stroke-width="2.2" fill="none" marker-end="url(#{mk})"/>')
        # card
        stroke = ACCENT_LT if c["accent"] else BASE
        fill = ACCENT_BG if c["accent"] else "#ffffff"
        p.append(f'<rect x="{CARD_X}" y="{top}" width="{CARD_R-CARD_X}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        tcol = ACCENT if c["accent"] else INK
        p.append(f'<text x="{inner_l}" y="{top+46}" font-size="28" font-weight="800" letter-spacing="2.4" fill="{tcol}">{c["kind"]}</text>')
        p.append(f'<text x="{q_x}" y="{top+45}" font-size="24" fill="{MUTED}">{esc(c["q"])}</text>')
        # bars
        by = top + 76
        if c["ordered"]:
            ay0, ay1 = by - 6, by + STEP * (len(c["rows"]) - 1) + BAR_H + 6
            p.append(f'<path d="M{CARD_X+12},{ay0} L{CARD_X+12},{ay1}" stroke="{ACCENT}" stroke-width="2.2" fill="none" marker-end="url(#ab)"/>')
        bcol = ACCENT if c["accent"] else "#8a8a85"
        for r, (lab, v) in enumerate(c["rows"]):
            yr = by + r * STEP
            p.append(f'<text x="{inner_l}" y="{yr+23}" font-size="24" fill="{INK}">{esc(lab)}</text>')
            p.append(f'<rect x="{track_x}" y="{yr}" width="{track_w}" height="{BAR_H}" rx="5" fill="#f1f1ef"/>')
            p.append(f'<rect x="{track_x}" y="{yr}" width="{max(track_w*v,3):.1f}" height="{BAR_H}" rx="5" fill="{bcol}"/>')
            p.append(f'<text x="{value_x}" y="{yr+23}" font-size="24" font-weight="700" fill="{INK}">{v:.2f}</text>')
        # note
        p.append(f'<text x="{inner_l}" y="{top+h-22}" font-size="22" fill="{MUTED}">{esc(c["note"])}</text>')
    p.append('</svg>')
    return "\n".join(p), H


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = str(int(time.time()))
    html = OUT / f"_fig_{stamp}.html"
    master = OUT / f"_master_{stamp}.png"
    svg, H = build()
    html.write_text(f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:#fff}}</style></head><body>{svg}</body></html>')
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
