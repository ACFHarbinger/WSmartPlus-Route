/**
 * Cypress runs the Studio in a plain browser, where the Tauri runtime is
 * absent. Stub the v2 IPC internals before the app loads so `invoke`/`listen`
 * resolve instead of crashing; commands return benign defaults.
 */

const IPC_DEFAULTS: Record<string, unknown> = {
  "plugin:event|listen": null,
  "plugin:event|unlisten": null,
  "plugin:store|load": 1,
  "plugin:store|get": [null, false],
  "plugin:store|set": null,
  "plugin:store|save": null,
  get_project_root: "/tmp/wsmart-e2e",
  list_files_recursive: [],
  read_text_file: "",
};

Cypress.on("window:before:load", (win) => {
  // Skip the first-run onboarding dialog and guided tour so specs can
  // interact with the shell directly.
  win.localStorage.setItem(
    "wsmart-studio-layout",
    JSON.stringify({
      state: { onboardingDismissed: true, guidedTourDismissed: true },
      version: 0,
    })
  );
  const w = win as unknown as Record<string, unknown>;
  w.__TAURI_EVENT_PLUGIN_INTERNALS__ = {
    unregisterListener: () => undefined,
  };
  w.__TAURI_INTERNALS__ = {
    invoke: (cmd: string) =>
      Promise.resolve(cmd in IPC_DEFAULTS ? IPC_DEFAULTS[cmd] : null),
    transformCallback: (cb: unknown) => {
      void cb;
      return Math.floor(Math.random() * 1e9);
    },
    metadata: { currentWindow: { label: "main" }, currentWebview: { label: "main" } },
    plugins: {},
  };
});

// The app intentionally logs IPC fallbacks in browser mode; don't fail on them.
Cypress.on("uncaught:exception", () => false);
