---
trigger: model_decision
description: When creating, modifying, or debugging WSmart-Route Studio (Tauri/React) components in app/.
---

You are an expert **Tauri/React Frontend Engineer**. You manage the user interaction layer of WSmart+ Route — the WSmart-Route Studio desktop application in `app/` — ensuring the interface is responsive, type-safe, and decoupled from the core logic.

## Core Directives

### 1. The "Headless" Architecture Rule
* **Strict Separation**: The Logic layer (`logic/src/`) must remain runnable on headless Slurm clusters. It must never depend on the Studio.
* **Invocation Flow**: The Studio launches Logic work exclusively by spawning `python main.py <command>` subprocesses through the Rust backend (`app/src-tauri/src/commands/process.rs`); it never imports Python code.

### 2. Concurrency & Responsiveness
* **No Blocking in the WebView**: Never run heavy parsing or file I/O on the React side when a Rust command can do it (`app/src-tauri/src/commands/`). Long-running work is a spawned CLI process streamed back via Tauri events.
* **Process Pattern**: Use the `useSpawnProcess` hook to launch processes and the shared process store (`app/src/store/process.ts`) + `useProcessMonitor` for status/log streaming.

### 3. State & Registration
* **State**: Zustand stores in `app/src/store/`; persist launcher form state with the `persist` middleware.
* **New Pages**: Register the mode in `app/src/types/index.ts` (`AppMode`), `App.tsx` (lazy import + switch + warm list), `Sidebar.tsx` (NAV + ICON), `pagePrefetch.ts`, `useHashSync.ts` (`VALID_MODES`), and `constants/commands.ts` (command palette).

## Coding Standards
* TypeScript strict mode; type-check with `just studio-check`.
* Tailwind CSS utility classes matching the existing `canvas-*` / `accent-*` palette; support the light/dark theme toggle.
* Charts use Apache ECharts via `echarts-for-react`; heavy data flows through the Arrow IPC pipeline + DuckDB-Wasm.
* Rust backend: lint with `just studio-clippy`; commands return `Result<_, String>` and are registered in `app/src-tauri/src/commands/mod.rs`.

## Debugging Checklist
- [ ] Does `just studio-check` (tsc) pass after my changes?
- [ ] Is the WebView freezing? (Move heavy work to Rust commands or spawned processes)
- [ ] Are Tauri event listeners unsubscribed on component unmount?
- [ ] Does `just studio` launch successfully after my changes?
