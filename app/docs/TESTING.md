# Testing the Studio

Two harnesses cover the frontend: **Vitest** (unit / component / integration,
jsdom) and **Cypress** (e2e in a real browser against `vite dev`).

## Vitest (`tests/`)

```
tests/
├── setup.ts          # jest-dom matchers + failing __TAURI_INTERNALS__ stub
├── unit/             # pure utils: pareto, symlog, chartLogScale,
│                     # processProgress, outputRunPath
├── component/        # Testing Library: KpiCard, StatusPill
└── integration/      # multi-module flows: gen data-engine pipeline
                      # (decode → filter → aggregate → ctx → pareto),
                      # global filter store ↔ chart transforms
```

Commands:

```bash
npm test                  # everything, once
npm run test:watch        # watch mode
npm run test:unit         # tests/unit only
npm run test:component    # tests/component only
npm run test:integration  # tests/integration only
npm run test:coverage     # v8 coverage (text + html)
```

Conventions:

- Tests import from `../../src/...` — no path aliases.
- The setup file installs a `__TAURI_INTERNALS__.invoke` stub that **throws**
  with the command name: unit tests must never reach real IPC; if one does,
  the failure names the offending command. Mock at the module boundary
  (`vi.mock("../../src/gen/io")`) when a data-layer function needs IPC.
- Component tests assert on rendered semantics (labels, class hooks like
  `.kpi-delta-pos`), not markup snapshots.

## Cypress (`cypress/`)

```
cypress/
├── e2e/
│   ├── navigation.cy.ts       # shell boot, sidebar nav, hash deep-links,
│   │                          # filter restore from hash
│   └── command_palette.cy.ts  # Ctrl+K open/filter/execute, Escape close
└── support/e2e.ts             # Tauri IPC stub + onboarding/tour suppression
```

Commands:

```bash
npm run test:e2e   # start vite dev, run all specs headless, shut down
npm run cy:open    # interactive runner (needs `npm run dev` in another shell)
npm run cy:run     # headless against an already-running dev server
```

How browser runs work without Tauri:

- `support/e2e.ts` installs `window.__TAURI_INTERNALS__` (and the event-plugin
  internals) before each page load; `invoke` resolves benign defaults from a
  per-command table (`IPC_DEFAULTS`) — extend that table when a new startup
  command needs a non-null answer.
- The persisted layout store is pre-seeded so the onboarding dialog and guided
  tour don't cover the UI.
- Cypress test isolation clears `localStorage` between tests; the seeding
  happens in `window:before:load`, so every test starts identically.

## Rust and Python tests

Rust command tests live under `src-tauri/` (`cargo test`). The Python
framework's pytest suites live at the repository root (`justfile` targets);
see `../../docs/TESTING.md`.
