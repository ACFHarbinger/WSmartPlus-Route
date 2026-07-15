import { useEffect } from "react";
import { useAppStore } from "../store/app";
import { applyDomTheme, getSystemTheme } from "../utils/theme";

/** Keeps DOM `dark` class and `effectiveTheme` in sync when preference is `system`. */
export function useThemeSync(): void {
  const theme = useAppStore((s) => s.theme);
  const setEffectiveTheme = useAppStore((s) => s.setEffectiveTheme);

  useEffect(() => {
    if (theme !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const sync = () => {
      const effective = getSystemTheme();
      applyDomTheme(effective);
      setEffectiveTheme(effective);
    };
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, [theme, setEffectiveTheme]);
}
