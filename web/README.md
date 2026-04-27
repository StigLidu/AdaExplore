# AdaExplore demo page

Static HTML/CSS/JS that visualizes one MCTS search trajectory and the AdaExplore method overview. Built to be deployed on GitHub Pages — no build step, no backend.

## Run locally

From the repository root:

```bash
python -m http.server -d web 8000
```

Then open http://localhost:8000.

## Bundle a different run

`build_run_data.py` packs a single problem's `step_*_log.json` / `step_*.py` /
`step_*_metrics.json` into one JSON the page can load:

```bash
python web/build_run_data.py \
  experiments/MCTS_0120_KB-l2_50/2_83 \
  web/showcase/kb-l2-p83.json \
  --label "KernelBench Level 2 — Problem 83"
```

The page loads `web/showcase/kb-l2-p83.json` by default; edit `RUN_URL` in
`app.js` to point at a different file.

## Files

- `index.html` — page layout (hero / overview / interactive tree / citation)
- `style.css` — minimal documentation-style theme
- `app.js` — d3 v7 (CDN) tree + timeline + detail panel
- `build_run_data.py` — log directory → packed JSON
- `showcase/` — pre-bundled run JSONs shipped with the page
