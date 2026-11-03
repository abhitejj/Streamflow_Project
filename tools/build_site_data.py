#!/usr/bin/env python3
"""Build the website's chart data and the standalone fullscreen chart pages.

Source of truth: the Plotly widgets written by the modelling pipeline, kept
unmodified in ``visualizations/source/``. This script

1. decodes every figure (including Plotly's base64 ``bdata`` arrays),
2. writes the numbers the site needs to ``assets/figdata.js`` (no values are
   changed; PR curves are thinned to ~400 points per curve for page weight),
3. writes one clean standalone page per chart to ``visualizations/*.html``
   (same file names as before, so existing links keep working).

When the pipeline regenerates its widgets, drop the new files into
``visualizations/source/`` and rerun:  python3 tools/build_site_data.py
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "visualizations" / "source"
OUT_JS = ROOT / "assets" / "figdata.js"
VIZ = ROOT / "visualizations"

_dec = json.JSONDecoder()


def _arr(v):
    """Decode Plotly typed-array payloads into plain lists."""
    if isinstance(v, dict) and "bdata" in v:
        a = np.frombuffer(base64.b64decode(v["bdata"]), dtype=v["dtype"])
        if "shape" in v:
            a = a.reshape([int(s) for s in str(v["shape"]).split(",")])
        return a.tolist()
    return v


def load_fig(name: str) -> dict:
    s = (SRC / f"{name}.html").read_text()
    i = s.index("Plotly.newPlot(")
    data, end = _dec.raw_decode(s, s.index("[", i))
    layout, _ = _dec.raw_decode(s, s.index("{", end))
    for t in data:
        for k in ("x", "y", "customdata"):
            if k in t:
                t[k] = _arr(t[k])
    return {"data": data, "layout": layout}


def r3(seq):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 3) for v in seq]


def nse_from_label(name: str):
    m = re.search(r"NSE:\s*(-?[\d.]+)", name)
    return float(m.group(1)) if m else None


def hydrographs() -> dict:
    out: dict = {}
    for st in ("karad", "kantamal"):
        out[st] = {}
        for h in ("1w", "4w"):
            f = load_fig(f"interactive_hydrograph_{st}_h{h}")
            obs, m1, m2 = f["data"]
            shapes = f["layout"]["shapes"]
            out[st][h] = {
                "x": [x[:10] for x in obs["x"]],
                "obs": r3(obs["y"]),
                "m1": r3(m1["y"]),
                "m2": r3(m2["y"]),
                "q95": round(shapes[0]["y0"], 3),
                "q90": round(shapes[1]["y0"], 3),
                "nse_m1": nse_from_label(m1["name"]),
                "nse_m2": nse_from_label(m2["name"]),
            }
    return out


def pr_curves() -> dict:
    f = load_fig("interactive_pr_curves_severe_drought")
    out: dict = {}
    for t in f["data"]:
        h = "2w" if t.get("xaxis") in (None, "x") else "4w"
        out.setdefault(h, {})
        if t["name"].startswith("No-Skill"):
            out[h]["noskill"] = round(t["y"][0], 3)
            continue
        key = "m1" if t["name"].startswith("M1") else "m2"
        n = len(t["x"])
        idx = np.unique(np.r_[np.linspace(0, n - 1, 400).round().astype(int), [0, n - 1]])
        cd = np.asarray(t["customdata"], dtype=float)
        out[h][key] = {
            "auc": float(re.search(r"PR-AUC:\s*([\d.]+)", t["name"]).group(1)),
            "recall": r3(np.asarray(t["x"])[idx]),
            "precision": r3(np.asarray(t["y"])[idx]),
            "cutoff": r3(cd[idx, 0]),
            "far": r3(cd[idx, 1]),
        }
    return out


def kge() -> dict:
    f = load_fig("interactive_kge_component_decomposition")
    comp = {"x": "r", "x2": "alpha", "x3": "beta"}
    out: dict = {"h": f["data"][0]["x"]}
    for t in f["data"]:
        if t["name"].startswith("Ideal"):
            continue
        key = {"M0": "m0", "M1": "m1", "M2": "m2"}[t["name"][:2]]
        out.setdefault(comp[t.get("xaxis", "x")], {})[key] = r3(t["y"])
    return out


def feature_importance() -> dict:
    f = load_fig("interactive_feature_importance_shift")
    a, b = f["data"]
    return {"features": a["y"], "h1": r3(a["x"]), "h8": r3(b["x"])}


def transfer() -> dict:
    f = load_fig("interactive_spatial_transfer_grouped_bars")
    out: dict = {}
    for t in f["data"]:
        h = "4w" if t.get("xaxis") in (None, "x") else "8w"
        d = out.setdefault(h, {})
        parts = [[p for p in re.split(r"<[^>]+>", x) if p.strip()] for x in t["x"]]
        d["basins"] = [p[0] for p in parts]
        d["states"] = [p[-1] for p in parts]
        key = {"M0": "m0", "M1": "m1", "M2": "m2"}[t["name"][:2]]
        d[key] = r3(t["y"])
    return out


PAGES = {
    "interactive_hydrographs_dashboard": ("Observed vs forecast hydrographs", "hydro", "Karad and Kantamal, 1-week and 4-week lead, test block 2016–2025."),
    "interactive_hydrograph_karad_h1w": ("Hydrograph · Karad · 1-week lead", "hydro:karad:1w", "Out-of-sample test block 2016–2025."),
    "interactive_hydrograph_karad_h4w": ("Hydrograph · Karad · 4-week lead", "hydro:karad:4w", "Out-of-sample test block 2016–2025."),
    "interactive_hydrograph_kantamal_h1w": ("Hydrograph · Kantamal · 1-week lead", "hydro:kantamal:1w", "Out-of-sample test block 2016–2025."),
    "interactive_hydrograph_kantamal_h4w": ("Hydrograph · Kantamal · 4-week lead", "hydro:kantamal:4w", "Out-of-sample test block 2016–2025."),
    "interactive_pr_curves_severe_drought": ("Severe drought (Q95) precision–recall", "pr", "Hover a curve to read the discharge cutoff and false alarm ratio."),
    "interactive_kge_component_decomposition": ("KGE component decomposition", "kge", "Correlation r, variability ratio α and bias ratio β; 1.0 is ideal."),
    "interactive_feature_importance_shift": ("Feature importance · 1-week vs 8-week lead", "fi", "Relative importance (%) of the 12 features shown, Karad test fold."),
    "interactive_spatial_transfer_grouped_bars": ("Zero-shot transfer to held-out basins", "transfer", "NSE on basins excluded from training."),
}

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — ES 335 Hydro-ML</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/site.css">
</head>
<body class="standalone">
<main class="standalone-wrap">
  <a class="back-link" href="../index.html#results">&larr; Back to the project page</a>
  <h1 class="standalone-title">{title}</h1>
  <p class="standalone-sub">{sub}</p>
  <div class="chart-card">
    <div data-chart="{chart}" class="chart chart-lg"></div>
  </div>
</main>
<script src="https://cdn.plot.ly/plotly-basic-2.35.2.min.js"></script>
<script src="../assets/figdata.js"></script>
<script src="../assets/benchmark-data.js"></script>
<script src="../assets/charts.js"></script>
<script>Charts.mountAll(document, {{eager: true}});</script>
</body>
</html>
"""


def main() -> None:
    fig = {
        "hydro": hydrographs(),
        "pr": pr_curves(),
        "kge": kge(),
        "fi": feature_importance(),
        "transfer": transfer(),
    }
    OUT_JS.parent.mkdir(exist_ok=True)
    OUT_JS.write_text(
        "/* Generated by tools/build_site_data.py from visualizations/source/*.html. Do not edit by hand. */\n"
        "window.FIGDATA = " + json.dumps(fig, separators=(",", ":")) + ";\n"
    )
    print(f"wrote {OUT_JS.relative_to(ROOT)} ({OUT_JS.stat().st_size / 1024:.0f} KB)")
    for name, (title, chart, sub) in PAGES.items():
        (VIZ / f"{name}.html").write_text(TEMPLATE.format(title=title, chart=chart, sub=sub))
    print(f"wrote {len(PAGES)} standalone chart pages to visualizations/")


if __name__ == "__main__":
    main()
