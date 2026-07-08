import { create } from "zustand";
import { persist } from "zustand/middleware";

interface LayoutState {
  sidebarOpen: boolean;
  shortcutsOpen: boolean;
  commandPaletteOpen: boolean;
  onboardingDismissed: boolean;
  guidedTourOpen: boolean;
  guidedTourStep: number;
  guidedTourDismissed: boolean;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setShortcutsOpen: (open: boolean) => void;
  setCommandPaletteOpen: (open: boolean) => void;
  setOnboardingDismissed: (dismissed: boolean) => void;
  setGuidedTourOpen: (open: boolean) => void;
  setGuidedTourStep: (step: number) => void;
  setGuidedTourDismissed: (dismissed: boolean) => void;
}

export const useLayoutStore = create<LayoutState>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      shortcutsOpen: false,
      commandPaletteOpen: false,
      onboardingDismissed: false,
      guidedTourOpen: false,
      guidedTourStep: 0,
      guidedTourDismissed: false,
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
      setShortcutsOpen: (shortcutsOpen) => set({ shortcutsOpen }),
      setCommandPaletteOpen: (commandPaletteOpen) => set({ commandPaletteOpen }),
      setOnboardingDismissed: (onboardingDismissed) => set({ onboardingDismissed }),
      setGuidedTourOpen: (guidedTourOpen) => set({ guidedTourOpen }),
      setGuidedTourStep: (guidedTourStep) => set({ guidedTourStep }),
      setGuidedTourDismissed: (guidedTourDismissed) => set({ guidedTourDismissed }),
    }),
    {
      name: "wsmart-studio-layout",
      partialize: (s) => ({
        sidebarOpen: s.sidebarOpen,
        onboardingDismissed: s.onboardingDismissed,
        guidedTourDismissed: s.guidedTourDismissed,
      }),
    }
  )
);
