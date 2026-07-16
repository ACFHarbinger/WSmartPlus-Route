export type ThemePreference = "dark" | "light" | "system";
export type EffectiveTheme = "dark" | "light";

export function getSystemTheme(): EffectiveTheme {
  if (typeof window === "undefined") return "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function resolveEffectiveTheme(preference: ThemePreference): EffectiveTheme {
  if (preference === "system") return getSystemTheme();
  return preference;
}

export function applyDomTheme(effective: EffectiveTheme): void {
  if (effective === "dark") {
    document.documentElement.classList.add("dark");
  } else {
    document.documentElement.classList.remove("dark");
  }
}

const THEME_CYCLE: ThemePreference[] = ["dark", "light", "system"];

export function nextThemePreference(current: ThemePreference): ThemePreference {
  const idx = THEME_CYCLE.indexOf(current);
  return THEME_CYCLE[(idx + 1) % THEME_CYCLE.length];
}
