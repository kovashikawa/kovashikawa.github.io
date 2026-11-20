# /// script
# dependencies = ["pillow"]
# ///
"""RL families diagram for the RL-for-language-models post.

Run: uv run tools/rl_post_families_diagram.py
Renders HTML/SVG with headless Chrome at 3x, downscales with LANCZOS (html-to-image-rendering pipeline).
"""
import pathlib, subprocess, time
from PIL import Image

OUT = pathlib.Path(__file__).parent.parent / "assets" / "images" / "rl-for-language-models"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
INK, MUTED, LIGHT, RULE, BASE = "#1E1E1E", "#555555", "#888888", "#e6e6e3", "#c9c9c4"
ACCENT, ACCENT_LT, ACCENT_BG = "#2a78d6", "#86b6ef", "#f0f6fd"

W = 1600
NH, GAP_IN, GAP_GRP = 66, 14, 38
TOP = 92
ROOT_X, ROOT_W = 40, 290
GRP_X, GRP_W = 430, 310
M_X, M_W = 850, 470
LIN_X = M_X + M_W

groups = [
    ("Human comparisons", "A reward model, or an implicit one", False, [
        ("DPO", "Implicit reward, supervised loss on preference pairs"),
        ("RLHF", "Reward model fit to comparisons, PPO with KL"),
    ]),
    ("An LLM as labeler or judge", "The labeler is a model, not a person", False, [
        ("RLAIF, Constitutional AI", "AI-labeled preferences, or a direct LLM reward"),
        ("Rubric rewards (RaR)", "LLM-judged weighted checklist, GRPO"),
    ]),
    ("A checker or verifier", "Right or wrong, programmed or learned", False, [
        ("Outcome and process rewards", "Learned verifier of answer or steps; mostly reranks"),
        ("Execution feedback (RLEF)", "Run the code against tests, over several turns"),
        ("RLVR", "Deterministic checker; PPO, GRPO, RLOO, Dr. GRPO"),
    ]),
    ("A proper scoring rule", "Confidence is part of the reward", True, [
        ("RLCR", "Checker plus a Brier term on stated confidence"),
        ("RLCD recipes", "Disclosed by others; calibration loss on the whole distribution"),
        ("System One: Jev", "Typed probabilities out; reward not published"),
    ]),
]
lineage = [  # (from, to, label, accent)
    ("RLHF", "RLAIF, Constitutional AI", "the labeler\nbecomes a model", False),
    ("RLVR", "RLCR", "adds a confidence\nterm to the reward", True),
    ("RLCR", "RLCD recipes", "related, not derived;\nscores the distribution", False),
    ("RLCD recipes", "System One: Jev", "same name, recipe\nnot published", False),
]
DASHED = {"System One: Jev"}

# ---- layout
y = TOP
pos, gpos = {}, []
for gname, gsub, acc, ms in groups:
    ys = []
    for name, sub in ms:
        pos[name] = (y, sub, acc)
        ys.append(y + NH / 2)
        y += NH + GAP_IN
    y += GAP_GRP - GAP_IN
    gpos.append(sum([ys[0], ys[-1]]) / 2)
H = int(y + 70)
root_cy = (gpos[0] + gpos[-1]) / 2


def curve(x1, y1, x2, y2, color, w=1.8):
    mx = (x1 + x2) / 2
    return f'<path d="M{x1} {y1} C{mx} {y1} {mx} {y2} {x2} {y2}" fill="none" stroke="{color}" stroke-width="{w}"/>'


svg = []
# column labels
for x, t in [(ROOT_X, "THE OBJECTIVE"), (GRP_X, "THE REWARD IS BUILT FROM"), (M_X, "METHOD AND WHAT IT DOES")]:
    svg.append(f'<text x="{x}" y="46" class="col">{t}</text>')

# edges first (under nodes)
for (gname, gsub, acc, ms), gcy in zip(groups, gpos):
    c = ACCENT_LT if acc else BASE
    svg.append(curve(ROOT_X + ROOT_W, root_cy, GRP_X, gcy, c))
    for name, _ in ms:
        ty = pos[name][0] + NH / 2
        if name in DASHED:
            svg.append(curve(GRP_X + GRP_W, gcy, M_X, ty, BASE).replace("/>", ' stroke-dasharray="6 4"/>'))
        else:
            svg.append(curve(GRP_X + GRP_W, gcy, M_X, ty, c))

# lineage arrows
for a, b, label, acc in lineage:
    ya, yb = pos[a][0] + NH / 2, pos[b][0] + NH / 2
    col = ACCENT if acc else MUTED
    bulge = 58
    svg.append(
        f'<path d="M{LIN_X} {ya} C{LIN_X+bulge} {ya} {LIN_X+bulge} {yb} {LIN_X+4} {yb}" fill="none" '
        f'stroke="{col}" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#ah{"a" if acc else "g"})"/>')
    l1, l2 = label.split("\n")
    my = (ya + yb) / 2
    svg.append(f'<text x="{LIN_X+bulge+16}" y="{my-4:.0f}" class="lin" fill="{col}">{l1}</text>')
    svg.append(f'<text x="{LIN_X+bulge+16}" y="{my+16:.0f}" class="lin" fill="{col}">{l2}</text>')

# root
rh = 112
svg.append(f'<rect x="{ROOT_X}" y="{root_cy-rh/2}" width="{ROOT_W}" height="{rh}" rx="10" fill="{INK}"/>')
svg.append(f'<text x="{ROOT_X+22}" y="{root_cy-14}" class="rt">One objective</text>')
svg.append(f'<text x="{ROOT_X+22}" y="{root_cy+14}" class="rs">max E[R] − β KL(π ‖ π_ref)</text>')
svg.append(f'<text x="{ROOT_X+22}" y="{root_cy+38}" class="rs2">a lens, not the algorithms</text>')

# groups
GH = 74
for (gname, gsub, acc, ms), gcy in zip(groups, gpos):
    s, f = (ACCENT, ACCENT_BG) if acc else (BASE, "#ffffff")
    svg.append(f'<rect x="{GRP_X}" y="{gcy-GH/2}" width="{GRP_W}" height="{GH}" rx="10" fill="{f}" stroke="{s}" stroke-width="1.8"/>')
    svg.append(f'<text x="{GRP_X+20}" y="{gcy-4}" class="gt" fill="{ACCENT if acc else INK}">{gname}</text>')
    svg.append(f'<text x="{GRP_X+20}" y="{gcy+20}" class="gs">{gsub}</text>')

# methods
for name, (py, sub, acc) in pos.items():
    s, f = (ACCENT, ACCENT_BG) if acc else (BASE, "#ffffff")
    dash = ' stroke-dasharray="7 5"' if name in DASHED else ""
    if name in DASHED:
        s, f = ACCENT_LT, "#ffffff"
    svg.append(f'<rect x="{M_X}" y="{py}" width="{M_W}" height="{NH}" rx="10" fill="{f}" stroke="{s}" stroke-width="1.8"{dash}/>')
    svg.append(f'<text x="{M_X+20}" y="{py+28}" class="mt" fill="{ACCENT if acc or name in DASHED else INK}">{name}</text>')
    svg.append(f'<text x="{M_X+20}" y="{py+50}" class="ms">{sub}</text>')

# legend
ly = H - 30
svg.append(f'<rect x="{ROOT_X}" y="{ly-13}" width="18" height="18" rx="4" fill="{ACCENT_BG}" stroke="{ACCENT}" stroke-width="1.8"/>')
svg.append(f'<text x="{ROOT_X+28}" y="{ly+2}" class="leg">Reward includes calibration</text>')
svg.append(f'<path d="M{ROOT_X+300} {ly-4} H{ROOT_X+340}" stroke="{MUTED}" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#ahg)"/>')
svg.append(f'<text x="{ROOT_X+352}" y="{ly+2}" class="leg">Builds on, relates to, or shares a name with</text>')

svg.append(f'<rect x="{ROOT_X+720}" y="{ly-13}" width="18" height="18" rx="4" fill="#fff" stroke="{ACCENT_LT}" stroke-width="1.8" stroke-dasharray="4 3"/>')
svg.append(f'<text x="{ROOT_X+748}" y="{ly+2}" class="leg">Output contract documented, training not public</text>')

html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:#fff}}
svg{{display:block;font-family:-apple-system,"Helvetica Neue",Helvetica,Arial,sans-serif}}
.col{{font-size:18px;font-weight:700;letter-spacing:1.8px;fill:{MUTED}}}
.rt{{font-size:21px;font-weight:700;fill:#fff}}
.rs{{font-size:16px;fill:#e6e6e3}}
.rs2{{font-size:14px;fill:#a8a8a8}}
.gt{{font-size:20px;font-weight:700}}
.gs{{font-size:15px;fill:{MUTED}}}
.mt{{font-size:20px;font-weight:700}}
.ms{{font-size:15.5px;fill:{MUTED}}}
.lin{{font-size:15px;font-style:italic}}
.leg{{font-size:15px;fill:{MUTED}}}
</style></head><body>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
<marker id="ahg" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0 0 L8 4.5 L0 9 z" fill="{MUTED}"/></marker>
<marker id="ahga" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0 0 L8 4.5 L0 9 z" fill="{ACCENT}"/></marker>
<marker id="aha" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0 0 L8 4.5 L0 9 z" fill="{ACCENT}"/></marker>
</defs>
{chr(10).join(svg)}
</svg></body></html>"""
(OUT / "rl-families.html").write_text(html)

master = OUT / f"_master_{int(time.time())}.png"
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                "--force-device-scale-factor=3", f"--window-size={W},{H}",
                f"--screenshot={master}", f"file://{OUT/'rl-families.html'}?v={int(time.time())}"],
               check=True, capture_output=True)
img = Image.open(master).convert("RGB")
img = img.resize((W * 2, H * 2), Image.Resampling.LANCZOS)
img.save(OUT / "rl-families.png", optimize=True)
master.unlink()
(OUT / "rl-families.html").unlink()
print("size", img.size)
