# Developing the Studio

## Prerequisites

- Node.js ≥ 20 and npm
- Rust stable toolchain (for the Tauri core)
- Tauri 2.0 Linux system deps (webkit2gtk-4.1, libappindicator, …) — see the
  [Tauri prerequisites](https://tauri.app/start/prerequisites/)
- The Python environment of the parent repo (`uv sync --all-extras`) so
  launched subprocesses work

## Everyday commands

```bash
npm install            # once, or after dependency changes
npm run dev            # frontend only, http://localhost:5173 (browser mode:
                       # Tauri IPC unavailable — launchers/files disabled)
npm run tauri:dev      # full desktop app with the Rust core (hot reload)
npm run build          # tsc + vite production build (frontend bundle)
npm run tauri:build    # desktop bundles; tauri:build:linux → .deb + .AppImage
npm test               # Vitest suites (see docs/TESTING.md)
npm run test:e2e       # Cypress e2e
```

## Project conventions

- **TypeScript strict**; `tsc --noEmit` must pass (`npm run build` runs it).
- No path aliases in app code — imports are relative; keep files inside the
  logical subdirectories described in [ARCHITECTURE.md](ARCHITECTURE.md).
- Pages are lazy-loaded from `App.tsx`; heavy libs are split via
  `vite.config.ts` `manualChunks` (echarts, maplibre, deckgl, monaco, duckdb,
  r3f, sigma).
- New backend capability = new `#[tauri::command]` in
  `src-tauri/src/commands/`, registered in `lib.rs`, with a thin typed wrapper
  on the TS side (`gen/io.ts` or a `utils/` module).
- Long-running work goes through `spawn_python_process` + the process store,
  never a blocking `invoke`.
- Persisted UI state uses `zustand/persist`; bump/partialize carefully — keys
  are user-visible state across app updates.

## First run

The app asks for the **project root** (the directory containing `main.py`).
Launchers, file browsing and Python probing stay disabled until it is set
(Settings page, or the onboarding dialog).

## Releasing

`tauri.conf.json` + `updater.example.json` configure the updater artefacts;
`npm run tauri:build:linux` produces the `.deb`/`.AppImage` bundles consumed by
the packager (`ci/packager.py`) and the Docker image (`docker/Dockerfile`).
