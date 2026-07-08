import { create } from "zustand";
import { persist } from "zustand/middleware";

interface LayoutState {
  sidebarOpen: boolean;
  shortcutsOpen: boolean;
  commandPaletteOpen: boolean;
  onboardingDismissed: boolean;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setShortcutsOpen: (open: boolean) => void;
  setCommandPaletteOpen: (open: boolean) => void;
  setOnboardingDismissed: (dismissed: boolean) => void;
}

export const useLayoutStore = create<LayoutState>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      shortcutsOpen: false,
      commandPaletteOpen: false,
      onboardingDismissed: false,
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
      setShortcutsOpen: (shortcutsOpen) => set({ shortcutsOpen }),
      setCommandPaletteOpen: (commandPaletteOpen) => set({ commandPaletteOpen }),
      setOnboardingDismissed: (onboardingDismissed) => set({ onboardingDismissed }),
    }),
    {
      name: "wsmart-studio-layout",
      partialize: (s) => ({ sidebarOpen: s.sidebarOpen, onboardingDismissed: s.onboardingDismissed }),
    }
  )
);
