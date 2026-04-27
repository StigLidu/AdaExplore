/* eslint-disable */
// AdaExplore search-tree demo. Vanilla d3 v7, no build step.
//
// Loads a single run.json (built by web/build_run_data.py) and animates the
// MCTS tree growing one node per step, while the right panel shows code +
// metrics for the selected (or most-recent) node, and a small timeline at the
// bottom plots best fast_p over steps.

(async function main() {
  const MANIFEST_URL = "showcase/index.json";

  const manifest = await fetch(MANIFEST_URL).then(r => r.json());
  const showcases = manifest.showcases || [];
  if (!showcases.length) throw new Error("manifest has no showcases");
  populatePicker(showcases);

  const initial = showcases.find(s => s.default) || showcases[0];
  await loadShowcase(initial);
})().catch(err => {
  console.error(err);
  const blurb = document.getElementById("run-blurb");
  if (blurb) blurb.textContent = "Failed to load showcase data — see browser console.";
});

function populatePicker(showcases) {
  const sel = document.getElementById("run-select");
  sel.innerHTML = "";
  for (const s of showcases) {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = s.label;
    if (s.default) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.disabled = false;
  sel.addEventListener("change", () => {
    const picked = showcases.find(s => s.id === sel.value);
    if (picked) loadShowcase(picked);
  });
}

async function loadShowcase(showcase) {
  const sel = document.getElementById("run-select");
  sel.disabled = true;
  document.getElementById("run-blurb").textContent = showcase.blurb || "";
  try {
    stopTimer();
    const data = await fetch(showcase.file).then(r => r.json());
    if (state.data) {
      // Switching: do a soft reset of the existing layout before initPage.
      state.treeSvg.select(".links").selectAll("path").remove();
      state.treeSvg.select(".nodes").selectAll("g.node").remove();
    }
    initPage(data);
  } finally {
    sel.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// State

const state = {
  data: null,            // loaded run JSON
  curStep: 0,            // index into data.steps shown so far (inclusive)
  selected: 0,           // which node_id is open in the detail panel
  playing: false,
  speed: 1200,           // ms per step (lower = faster)
  timer: null,
  saturationShown: false, // true once we've auto-paused at the 10× milestone
  viewBoxScale: 1,        // SVG units per CSS pixel; set each redraw from viewBox
  milestoneSet: new Set(),// node_ids that pushed global_best higher (prefix-max)
  // d3 selections / accessors filled in below
  treeSvg: null,
  tlSvg: null,
};

// ---------------------------------------------------------------------------
// Page bootstrap

let chromeReady = false;

function initPage(data) {
  state.data = data;
  document.getElementById("max-step").textContent = data.total_steps;

  // Pre-compute the prefix-max milestones: node_ids that strictly improved
  // global_best at the moment they appeared. Only these get text labels in
  // the tree (everyone else is unlabeled to avoid clutter).
  state.milestoneSet = new Set();
  let prevBest = 0;
  for (const s of data.steps) {
    if (s.global_best_fast_p > prevBest + 1e-9) {
      state.milestoneSet.add(s.node_id);
      prevBest = s.global_best_fast_p;
    }
  }

  renderHeaderMetrics();

  // SVG containers and event listeners only need to be set up once. Subsequent
  // showcase loads just refresh the data and call redraw().
  if (!chromeReady) {
    setupTreeSvg();
    setupTimelineSvg();
    setupControls();
    chromeReady = true;
  }

  // Reset playback state so a switch always starts clean from the dummy root.
  togglePlay(false);
  state.curStep = 0;
  state.selected = 0;
  state.saturationShown = false;
  document.getElementById("saturation-banner").hidden = true;

  // Auto-pace: target roughly the same total wall-clock playback (~120s) for
  // both 50-step and 200-step runs. Clamp so the speed never feels jerky or
  // glacial. The slider sits at the matching position so the user can still
  // override.
  const totalSteps = data.total_steps;
  state.speed = Math.min(1200, Math.max(350, Math.round(120000 / totalSteps)));
  const slider = document.getElementById("speed");
  if (slider) slider.value = +slider.max + +slider.min - state.speed;

  redraw();

  // Auto-play after a short pause so the user sees the seed tree first.
  setTimeout(() => togglePlay(true), 600);
}

function renderHeaderMetrics() {
  const d = state.data;
  const steps = d.steps;
  const finalBest = steps[steps.length - 1].global_best_fast_p;
  const firstCorrect = steps.find(s => s.score.correct);
  const baseline = steps.find(s => s.baseline_time_us)?.baseline_time_us;
  const hw = steps.find(s => s.hardware)?.hardware || "—";

  const cells = [
    `<div>final best <strong>${finalBest.toFixed(2)}×</strong></div>`,
    `<div>first correct <strong>step ${firstCorrect ? firstCorrect.step : "—"}</strong></div>`,
    baseline ? `<div>baseline <strong>${baseline.toFixed(2)} ms</strong></div>` : "",
    `<div>hardware <strong>${hw}</strong></div>`,
  ].filter(Boolean);
  document.getElementById("metrics-strip").innerHTML = cells.join("");
}

// ---------------------------------------------------------------------------
// Tree

const TREE_MARGIN = { top: 28, right: 32, bottom: 32, left: 32 };

function setupTreeSvg() {
  const svg = d3.select("#tree");
  state.treeSvg = svg;

  const g = svg.append("g").attr("class", "tree-g");
  g.append("g").attr("class", "links");
  g.append("g").attr("class", "context-links");
  g.append("g").attr("class", "nodes");
}

function visibleSteps() {
  // Up to and including curStep (note: step 0 = dummy root, step N corresponds
  // to data.steps[N] since the array starts at the root).
  return state.data.steps.slice(0, state.curStep + 1);
}

function buildHierarchy() {
  const steps = visibleSteps();
  const byId = new Map();
  for (const s of steps) byId.set(s.node_id, s);

  const root = byId.get(0);
  if (!root) return null;
  // For each node, list its children that are visible.
  const childrenOf = new Map();
  for (const s of steps) {
    if (s.parent_node_id === null || s.parent_node_id === undefined) continue;
    if (!childrenOf.has(s.parent_node_id)) childrenOf.set(s.parent_node_id, []);
    childrenOf.get(s.parent_node_id).push(s);
  }
  function attach(node) {
    return {
      ...node,
      children: (childrenOf.get(node.node_id) || []).map(attach),
    };
  }
  return d3.hierarchy(attach(root));
}

function nodeColor(s) {
  if (!s) return "var(--c-pending)";
  if (s.created_by === "dummy_root") return "#cbd5da";
  if (!s.score.compiled) return "var(--c-fail)";
  if (!s.score.correct) return "var(--c-incorrect)";
  const fp = s.score.fast_p || 0;
  if (fp <= 0.5) return "var(--c-slow)";
  if (fp < 1.05) return "var(--c-slow)";
  if (fp < 2)    return "var(--c-fast-1)";
  if (fp < 5)    return "var(--c-fast-2)";
  return "var(--c-fast-3)";
}

function nodeRadius(s) {
  // The "natural" radius (in CSS pixels). compressed log10 so a single 80×
  // outlier doesn't visually crush everything else around it.
  // 1× ≈ 8.2, 5× ≈ 10.1, 10× ≈ 11.2, 50× ≈ 13.8, 100× ≈ 15. Capped at 16.
  let basePx;
  if (!s.score.correct) basePx = 5;
  else {
    const fp = Math.max(s.score.fast_p || 0, 0);
    basePx = Math.min(7 + Math.log10(1 + fp) * 4, 16);
  }
  // Convert pixel-base into SVG units using the current viewBox scale, so
  // the node looks the same screen size regardless of how zoomed-out the
  // viewBox is (which depends on how many nodes are visible).
  return basePx * state.viewBoxScale;
}

// Per-node spacing constants for the adaptive layout. These determine how much
// breathing room each tree node gets at maximum density.
const NODE_X_SPACING = 28;   // horizontal px per node at the widest depth
const LEVEL_Y_SPACING = 80;  // vertical px per depth level
const MIN_LAYOUT_W = 800;
const MIN_LAYOUT_H = 460;

function fullTreeDimensions() {
  // Use the FULL final tree (not just the visible-so-far subset) to choose the
  // layout size. That keeps the SVG canvas a stable size for the entire
  // animation — no jarring resize as new nodes appear.
  let maxDepth = 0;
  const breadthByDepth = new Map();
  for (const s of state.data.steps) {
    if (s.depth > maxDepth) maxDepth = s.depth;
    breadthByDepth.set(s.depth, (breadthByDepth.get(s.depth) || 0) + 1);
  }
  let maxBreadth = 1;
  for (const v of breadthByDepth.values()) if (v > maxBreadth) maxBreadth = v;
  return {
    layoutW: Math.max(MIN_LAYOUT_W, maxBreadth * NODE_X_SPACING),
    layoutH: Math.max(MIN_LAYOUT_H, (maxDepth + 1) * LEVEL_Y_SPACING),
  };
}

function drawTree() {
  const svg = state.treeSvg;
  const root = buildHierarchy();
  if (!root) return;

  // Layout uses dimensions sized for the FULL final tree. This gives every
  // node enough breathing room even at peak density; we then use viewBox to
  // fit only the *visible* subset into the viewport.
  const { layoutW, layoutH } = fullTreeDimensions();
  const layout = d3.tree().size([layoutW, layoutH]);
  layout(root);

  svg.select(".tree-g").attr("transform", null);

  // viewBox = bounding box of currently visible nodes + padding. The SVG
  // element stays at its CSS pixel size; the browser scales the viewBox into
  // it, so the visible subtree always fills the viewport without scrollbars.
  // As new nodes appear the viewBox grows smoothly (transitioned below).
  // Minimum dimensions prevent the early-state "two nodes in a tiny canvas"
  // case from blowing every node up to comically huge screen sizes.
  const xs = root.descendants().map(d => d.x);
  const ys = root.descendants().map(d => d.y);
  const PAD = 60;
  const MIN_VB_W = 700;
  const MIN_VB_H = 380;
  let vbX = Math.min(...xs) - PAD;
  let vbY = Math.min(...ys) - PAD * 0.7;
  let vbW = (Math.max(...xs) - Math.min(...xs)) + 2 * PAD;
  let vbH = (Math.max(...ys) - Math.min(...ys)) + 2 * PAD * 0.7;
  if (vbW < MIN_VB_W) { vbX -= (MIN_VB_W - vbW) / 2; vbW = MIN_VB_W; }
  if (vbH < MIN_VB_H) { vbY -= (MIN_VB_H - vbH) / 2; vbH = MIN_VB_H; }
  const newVB = [vbX, vbY, vbW, vbH];

  // Update the scale factor that nodeRadius() uses to keep on-screen node
  // size constant. Clamp so radii never grow huge or shrink to invisible.
  const svgPixelW = svg.node().clientWidth || 900;
  state.viewBoxScale = Math.max(0.6, Math.min(2.2, vbW / svgPixelW));

  const oldAttr = svg.attr("viewBox");
  const oldVB = oldAttr ? oldAttr.split(/\s+/).map(Number) : newVB;
  if (state.playing || oldAttr === null) {
    svg.transition().duration(state.playing ? 320 : 200)
      .attrTween("viewBox", () => {
        const i = d3.interpolateArray(oldVB, newVB);
        return t => i(t).join(" ");
      });
  } else {
    svg.attr("viewBox", newVB.join(" "));
  }

  const links = root.links();
  const nodes = root.descendants();

  const linkSel = svg.select(".links")
    .selectAll("path.link")
    .data(links, d => d.target.data.node_id);

  linkSel.exit().remove();

  linkSel.enter()
    .append("path")
    .attr("class", d => `link ${d.target.data.created_by === "large_step" ? "large" : "small"}`)
    .attr("d", d => {
      const sx = d.source.x ?? 0;
      const sy = d.source.y ?? 0;
      return d3.linkVertical()({ source: [sx, sy], target: [sx, sy] });
    })
    .merge(linkSel)
    .transition().duration(state.playing ? 300 : 250)
    .attr("class", d => `link ${d.target.data.created_by === "large_step" ? "large" : "small"}`)
    .attr("d", d => d3.linkVertical()({
      source: [d.source.x, d.source.y],
      target: [d.target.x, d.target.y],
    }));

  // Best node id at this point in time
  const bestId = state.data.steps[state.curStep].global_best_node_id;

  const nodeSel = svg.select(".nodes")
    .selectAll("g.node")
    .data(nodes, d => d.data.node_id);

  nodeSel.exit().remove();

  const nodeEnter = nodeSel.enter()
    .append("g")
    .attr("class", "node")
    .attr("transform", d => {
      // Enter from parent position so growth feels organic.
      const px = d.parent ? d.parent.x : d.x;
      const py = d.parent ? d.parent.y : d.y;
      return `translate(${px},${py})`;
    })
    .style("cursor", "pointer")
    .on("click", (_evt, d) => {
      state.selected = d.data.node_id;
      renderDetail();
      svg.selectAll("g.node").classed("selected", n => n.data.node_id === state.selected);
    });

  nodeEnter.append("circle")
    .attr("r", 0)
    .attr("fill", d => nodeColor(d.data));

  nodeEnter.append("text")
    .attr("dy", -2)
    .attr("text-anchor", "middle");

  // Highlight the context nodes that informed the currently-selected node.
  // Context relationships are not tree edges (a small_step's context is the
  // chain of recent edits in working memory; a large_step usually has none but
  // can pull a representative kernel from the pool), so we draw them as a
  // separate dashed amber overlay.
  const selectedDatum = nodes.find(n => n.data.node_id === state.selected);
  const contextIds = new Set(
    (selectedDatum && selectedDatum.data.context_node_ids) || []
  );

  // Update positions / styles on all nodes (entering + already-there).
  const merged = nodeEnter.merge(nodeSel);
  merged.classed("best", d => d.data.node_id === bestId);
  merged.classed("context", d => contextIds.has(d.data.node_id));
  merged.classed("milestone", d => state.milestoneSet.has(d.data.node_id));
  merged.transition().duration(state.playing ? 350 : 250)
    .attr("transform", d => `translate(${d.x},${d.y})`);

  merged.select("circle")
    .transition().duration(state.playing ? 350 : 250)
    .attr("r", d => nodeRadius(d.data))
    .attr("fill", d => nodeColor(d.data));

  // Only label milestone nodes (those that pushed global_best up) plus the
  // currently-selected node — keeps the dense tree readable.
  merged.select("text")
    .text(d => {
      const id = d.data.node_id;
      if (id === 0) return "";
      const labelIt = state.milestoneSet.has(id) || id === state.selected;
      if (!labelIt) return "";
      return d.data.score.correct ? `${d.data.score.fast_p.toFixed(2)}×` : "";
    })
    .attr("font-size", 13 * state.viewBoxScale)
    .transition().duration(state.playing ? 350 : 250)
    .attr("y", d => -nodeRadius(d.data) - 3 * state.viewBoxScale);

  merged.classed("selected", d => d.data.node_id === state.selected);

  // Dashed amber arcs from each context node to the selected node.
  const contextLinkData = (selectedDatum && contextIds.size > 0)
    ? nodes
        .filter(n => contextIds.has(n.data.node_id))
        .map(c => ({ source: c, target: selectedDatum }))
    : [];

  const ctxSel = svg.select(".context-links")
    .selectAll("path.context-link")
    .data(contextLinkData, d => `${d.source.data.node_id}->${d.target.data.node_id}`);

  ctxSel.exit().remove();

  ctxSel.enter()
    .append("path")
    .attr("class", "context-link")
    .attr("d", d => d3.linkVertical()({
      source: [d.source.x, d.source.y],
      target: [d.source.x, d.source.y],
    }))
    .merge(ctxSel)
    .transition().duration(state.playing ? 320 : 220)
    .attr("d", d => d3.linkVertical()({
      source: [d.source.x, d.source.y],
      target: [d.target.x, d.target.y],
    }));
}

// ---------------------------------------------------------------------------
// Timeline (best fast_p over steps)

function setupTimelineSvg() {
  state.tlSvg = d3.select("#timeline");
  state.tlSvg.append("g").attr("class", "tl-area-g");
  state.tlSvg.append("path").attr("class", "tl-area");
  state.tlSvg.append("path").attr("class", "tl-line");
  state.tlSvg.append("g").attr("class", "tl-axis tl-axis-x");
  state.tlSvg.append("g").attr("class", "tl-axis tl-axis-y");
  state.tlSvg.append("line").attr("class", "tl-cursor");
}

function drawTimeline() {
  const svg = state.tlSvg;
  const node = svg.node();
  const w = node.clientWidth || 900;
  const h = 120;
  const margin = { top: 8, right: 12, bottom: 22, left: 38 };
  const iw = w - margin.left - margin.right;
  const ih = h - margin.top - margin.bottom;

  const steps = state.data.steps;
  // Sqrt-scale x: emphasizes the early steps where most of the action happens
  // without over-stretching the very beginning the way log scale does.
  const N = steps.length - 1;
  const x = d3.scaleSqrt().domain([1, N]).range([margin.left, margin.left + iw]).clamp(true);
  const yMax = Math.max(1, d3.max(steps, d => d.global_best_fast_p) || 1);
  const y = d3.scaleLog().domain([1, yMax]).range([margin.top + ih, margin.top]).clamp(true);

  const line = d3.line()
    .defined(d => d.global_best_fast_p > 0)
    .x((_, i) => x(i))
    .y(d => y(Math.max(d.global_best_fast_p, 1)));
  const area = d3.area()
    .defined(d => d.global_best_fast_p > 0)
    .x((_, i) => x(i))
    .y0(margin.top + ih)
    .y1(d => y(Math.max(d.global_best_fast_p, 1)));

  svg.select(".tl-line").datum(steps).attr("d", line);
  svg.select(".tl-area").datum(steps).attr("d", area);

  // Custom tick values — round numbers spread well across the sqrt axis.
  // Filter to those <= N so 50-step runs don't render a "200" tick.
  const xTicks = [1, 5, 25, 50, 100, 200].filter(v => v <= N);
  if (xTicks[xTicks.length - 1] !== N) xTicks.push(N);
  svg.select(".tl-axis-x")
    .attr("transform", `translate(0,${margin.top + ih})`)
    .call(d3.axisBottom(x).tickValues(xTicks).tickFormat(d3.format("d")));

  svg.select(".tl-axis-y")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(4, "~g"));

  // cursor at current step
  svg.select(".tl-cursor")
    .attr("x1", x(state.curStep))
    .attr("x2", x(state.curStep))
    .attr("y1", margin.top)
    .attr("y2", margin.top + ih);
}

// ---------------------------------------------------------------------------
// Detail panel

function renderDetail() {
  const id = state.selected;
  const s = state.data.steps.find(x => x.node_id === id) || state.data.steps[0];
  const el = document.getElementById("detail");

  if (!s || s.node_id === 0) {
    el.innerHTML = `
      <h3>Root</h3>
      <p class="hint">The dummy root represents the initial PyTorch reference.
        Children of the root are full-kernel rewrites (large steps).</p>
      <pre><code>${escapeHtml(state.data.reference_src.slice(0, 1800))}</code></pre>
    `;
    return;
  }

  const pillType = s.created_by === "large_step"
    ? `<span class="pill large">large step</span>`
    : `<span class="pill small">small step</span>`;
  const verdict = s.score.compiled
    ? (s.score.correct
        ? `<span class="pill" style="background:#4ca85a22;color:#1f6e36">correct, ${s.score.fast_p.toFixed(2)}× speedup</span>`
        : `<span class="pill" style="background:#ef9c1f22;color:#9a6914">compiled, incorrect</span>`)
    : `<span class="pill" style="background:#d0474622;color:#a13231">compile failed</span>`;

  el.innerHTML = `
    <h3>Step ${s.step} ${pillType}</h3>
    <p style="margin:0 0 0.6rem">${verdict}</p>
    <dl>
      <dt>parent</dt><dd>step ${s.parent_node_id ?? "—"}</dd>
      <dt>depth</dt><dd>${s.depth}</dd>
      <dt>reward</dt><dd>${s.avg_reward.toFixed(3)}</dd>
      <dt title="Earlier nodes whose kernels were given to the LLM as working-memory / pool context for this proposal.">context</dt>
      <dd>${s.context_node_ids.length
            ? s.context_node_ids.map(id => `step ${id}`).join(", ")
            : "—"}</dd>
      ${s.runtime_us ? `<dt>runtime</dt><dd>${s.runtime_us.toFixed(2)} ms</dd>` : ""}
    </dl>
    ${s.error ? `<div class="err">${escapeHtml(s.error)}</div>` : ""}
    ${s.code ? `<pre><code>${escapeHtml(s.code.slice(0, 4000))}</code></pre>` : ""}
  `;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
  }[c]));
}

// ---------------------------------------------------------------------------
// Controls / playback

function setupControls() {
  document.getElementById("play-pause").addEventListener("click", () => togglePlay());
  document.getElementById("step-fwd").addEventListener("click", () => stepBy(+1));
  document.getElementById("step-back").addEventListener("click", () => stepBy(-1));
  document.getElementById("reset").addEventListener("click", () => reset());
  document.getElementById("banner-dismiss").addEventListener("click", () => {
    document.getElementById("saturation-banner").hidden = true;
    togglePlay(true);
  });

  const speed = document.getElementById("speed");
  speed.addEventListener("input", () => {
    // Slider value is "ms per step"; flip the slider so right = faster.
    const max = +speed.max;
    const min = +speed.min;
    state.speed = max + min - +speed.value;
    if (state.timer) restartTimer();
  });
}

function togglePlay(forceOn) {
  const want = forceOn === undefined ? !state.playing : !!forceOn;
  state.playing = want;
  const btn = document.getElementById("play-pause");
  btn.textContent = want ? "❚❚ Pause" : "▶ Play";
  if (want) restartTimer();
  else stopTimer();
}
function restartTimer() {
  stopTimer();
  state.timer = setInterval(() => stepBy(+1), state.speed);
}
function stopTimer() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
}
function stepBy(d) {
  const prevBest = state.data.steps[state.curStep].global_best_fast_p;
  state.curStep = Math.max(0, Math.min(state.data.steps.length - 1, state.curStep + d));
  const cur = state.data.steps[state.curStep];

  // Auto-select the node that just appeared on a forward step.
  if (d > 0) state.selected = cur.node_id;

  // Auto-pause once if we just crossed the 10× MCTS reward saturation point —
  // beyond this MCTS no longer differentiates speedups, so the demo halts to
  // explain why what comes next is exploration luck rather than search signal.
  if (state.playing && d > 0 && !state.saturationShown
      && prevBest < 10 && cur.global_best_fast_p >= 10) {
    state.saturationShown = true;
    document.getElementById("saturation-banner").hidden = false;
    togglePlay(false);
  } else if (state.playing && state.curStep >= state.data.steps.length - 1) {
    togglePlay(false);
  }

  redraw();
}
function reset() {
  togglePlay(false);
  state.curStep = 0;
  state.selected = 0;
  redraw();
}

function redraw() {
  document.getElementById("cur-step").textContent = state.curStep;
  drawTree();
  drawTimeline();
  renderDetail();
}
