# Brief — Opencode: interactive + 3D/4D visualisation for `docs/website/`

**Branch:** `feat/paper-results-and-website` · **Bus:** `.agent/bus/2026-08-25.md`
**Your lane:** `src/components/` (visualisation components), `src/hooks/`,
`src/data/`, any new `src/viz/`. **Not** `src/styles/` or `Layout.tsx` —
those are Agy's (`.agent/tasks/gemini-website-design.md`). Consume her tokens;
never hardcode a colour. Claim shared files on the bus first.

## The point

Right now the site *describes* the framework. It should *show* it running.
Everything below is backed by real data already in this repo — none of it
needs to be faked, and none of it may be.

## V1 — The simulator + policy pipeline, as a working diagram

A three-stage pipeline the visitor can step through:

```
mandatory selection  →  route construction  →  route improvement
   (32 strategies)        (8 families)           (33 improvers)
```

Source of truth: `logic/src/policies/{mandatory_selection,route_construction,
route_improvement}/`. Let the visitor pick one option per stage and see the
configuration they have built, with the real algorithm names. The paper calls
this the "policy configuration space" — there is a static rendering of it at
`assets/papers/.../Images/policy_configuration_space.png`. Beat it.

## V2 — Bin selection, animated

The most interesting thing the framework does is decide *which bins to skip
today*. Show it: a set of bins filling over simulated days, each with a fill
level, and the selection rule drawing the cut.

- **Last-Minute (CF70 / CF90)**: collect a bin when its *predicted end-of-day*
  fill crosses the threshold. Let the visitor drag the threshold and watch
  the selected set and the overflow count change.
- **Service-Level (SL1 / SL2)**: collect to keep overflow probability under a
  bound.
- **Look-Ahead (LA)**: decide using future periods, not just today.

The trade-off this reveals is the paper's central finding and it is real —
from the 480-run 30-day grid: LM-CF90 gets 7.74 kg/km with 28.6 overflows,
LM-CF70 gets 6.23 with 7.7, SL-SL1 5.65 with 6.3, SL-SL2 4.25 with 9.3.
Collect later, haul more per kilometre, overflow more. If the visitor's
intuition after playing with your widget is that ordering, it works.

## V3 — Routing, 3D/4D

The "4D" that matters here is genuinely present in the data: a
**multi-period** problem is 3D space plus time. Real coordinates and real
routes are in `assets/output/30days/**/log_*.json` and the distance matrices
(`gmaps_distmat.csv`, `osm_distmat.csv`); the 317-bin map figure is at
`assets/papers/.../Images/map317.png`.

Build a route view where the third axis or the animation axis is the day:
30 days of routes over Rio Maior (N=100/170) or Figueira da Foz (N=350),
scrubbable. Watch which bins get revisited and which get skipped for a week.
That is the multi-period structure made visible, and no static figure in the
paper conveys it.

`three` + `@react-three/fiber` + `@react-three/drei` are the reasonable choice
(the reference site uses them). Add them to `docs/website/package.json`
properly. **Budget: keep the 3D chunk lazy-loaded and code-split** — it must
not land in the initial bundle.

## V4 — Results charts with a point of view

Replace generic charts with ones that carry the argument:

- A **Pareto view** of kg/km against overflows, coloured by constructor,
  with the front drawn. `logic/gen/gen_simulation_analysis.py` already builds
  this in matplotlib — read it for the exact semantics before reimplementing.
- A **paired improver comparison**: CLS vs Fast-TSP on the 240 configurations
  that differ only in improver. CLS wins 212 of them.
- A **constructor × scenario heatmap** with the honest caveats built in.

**Data integrity — non-negotiable.** Two rows in
`public/global/simulation/simulation_summary.csv` are a truncated run
(`SWC-TCF / LA / Gamma-3 / N=350`, `days=15` of 30, ~half the tonnage of
every peer). Exclude them from aggregates or label them; do not silently
average them in. And the 90-day CSV is *not* a complete grid — ALNS has zero
runs at 90 days — so any 90-day comparison must be scoped to what was
actually run. Read today's bus entry in full before writing a chart.

## Ground rules

- Every visualisation reads real repo data. If you need it reshaped for the
  web, write a generator under `logic/gen/` that emits JSON into
  `docs/website/public/data/` — do not hand-copy numbers into a `.ts` file.
- `prefers-reduced-motion` must disable autoplay and continuous animation.
- Every canvas/WebGL view needs a non-WebGL fallback and a text description.
- Keyboard reachable: if it can be dragged, it can be arrowed.
- `npm run build` and `npm run lint` pass; report the bundle delta.
- `docs/moon/CHANGELOG.md` in the same commit. Never push to `main`.
- Codex reviews your diff before it counts as done.
