import { create } from "zustand";
import { persist } from "zustand/middleware";

interface LayoutState {
  sidebarOpen: boolean;
  shortcutsOpen: boolean;
  commandPaletteOpen: boolean;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setShortcutsOpen: (open: boolean) => void;
  setCommandPaletteOpen: (open: boolean) => void;
}

export const useLayoutStore = create<LayoutState>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      shortcutsOpen: false,
      commandPaletteOpen: false,
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
      setShortcutsOpen: (shortcutsOpen) => set({ shortcutsOpen }),
      setCommandPaletteOpen: (commandPaletteOpen) => set({ commandPaletteOpen }),
    }),
    {
      name: "wsmart-studio-layout",
      partialize: (s) => ({ sidebarOpen: s.sidebarOpen }),
    }
  )
);
