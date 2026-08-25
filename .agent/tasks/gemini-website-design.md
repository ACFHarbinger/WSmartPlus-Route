# Brief — Agy / Gemini: visual identity and design system for `docs/website/`

**Branch:** `feat/paper-results-and-website` · **Bus:** `.agent/bus/2026-08-25.md`
**Your lane:** `src/styles/`, `src/components/Layout.tsx`, page *layout and
chrome*, `public/` assets. **Not** the interactive/3D components — those are
Opencode's (`.agent/tasks/opencode-website-interactive.md`). Coordinate on the
bus before touching a file the other one owns.

## What exists

A 2,000-line React 19 + Vite + react-router site. Seven pages (Home, Platform,
Research, Benchmarks, Studio, Docs, Roadmap), one `Layout`, three
visual components (`RouteGraph`, `RouteAtlas`, `NetworkField`), and an
853-line `site.css`. The current look is the default "AI startup" treatment:
three blurred gradient orbs on a dark background, glass nav, generic
sans-serif. It is competent and completely anonymous. That is the problem.

## The bar

The reference for *quality* is
`/home/pkhunter/Repositories/Repo/Image-Toolkit/docs/website`. Study it —
`src/styles/tokens.css`, `src/styles/theme.css`, `Hero3D.tsx`,
`components/journal/`. Note how it commits to one specific idea (a
photographic darkroom / "Optic Lab": ink and paper, silver, optic cyan and
magenta, review amber, hairline rules, restrained motion) and then applies it
everywhere without exception. Every colour is a named semantic token; nothing
is invented per-page.

**Match that level of craft. Do not match that look.** If a visitor could see
both sites and think they came from the same template, the work has failed.
Concretely, off-limits as borrowings: the ink/paper/silver palette, the amber
accent, the cherry-blossom motif, the journal/darkroom framing, the hero
treatment.

## The idea to commit to

WSmart+ Route is about municipal waste collection routing — trucks, bins,
streets, days of the week, a city that has to be serviced whether or not the
optimiser is clever. The aesthetic that belongs to it is **cartographic and
civic**, not laboratory: survey maps and contour lines, transit-diagram
geometry, municipal signage typography, the visual language of an
infrastructure operations room. Route polylines, fill-level gauges, a
calendar-like multi-period rhythm.

That is a strong recommendation, not an order. If you can argue for something
better on the bus, argue for it — but pick *one* idea and commit to it as
completely as the reference site commits to its own.

## Deliverables

1. **A token system** — `src/styles/tokens.css`, semantic names, in the same
   spirit as the reference but with your own palette. Light and dark must both
   be real, designed states, not one inverted.
2. **Typography** — a type scale and a font pairing with genuine character.
   Google Fonts only (nothing else will load). Give every face a real fallback
   stack.
3. **Re-skin all seven pages and the `Layout` chrome** against the tokens.
   No page may hardcode a colour.
4. **A distinctive hero** for Home that is *not* three blurred orbs.
5. **Motion** — deliberate, and every animation must respect
   `prefers-reduced-motion` (the reference does this at the token level; copy
   the technique, not the values).
6. **Responsive down to 360px.** No horizontal body scroll at any width; wide
   content scrolls inside its own container.
7. **Accessibility** — 4.5:1 contrast on body text in both themes, visible
   focus rings, real landmarks.

## Content that should drive the design

Do not design against lorem ipsum. The site should surface:

- The **policy pipeline**: mandatory selection → route construction → route
  improvement. Three stages, 32 selection strategies, 8 constructor families,
  33 improvers. That structure is the site's spine.
- The **benchmark results** — `public/global/simulation/simulation_summary.csv`
  is real, 480 runs. Read today's bus entry for what it actually says before
  designing a page about it.
- The **paper and presentation** —
  `assets/papers/Simulation_Framework_for_the_MPVRP_with_Profits_in_Smart_Waste_Collection/`.

## Ground rules

- `docs/moon/CHANGELOG.md` updated in the same commit as the work.
- Commit incrementally, on the shared branch. Never push to `main`.
- `npm run build` and `npm run lint` must pass before you call anything done.
- Post to the bus when you claim a file, and when you finish.
- Codex reviews your diff before it counts as done.
