import { useEffect } from "react";
import { useGlobalFileDrop } from "../../hooks/useGlobalFileDrop";
import { useLayoutStore } from "../../store/layout";
import { CommandPalette } from "./CommandPalette";
import { KeyboardShortcutsHelp } from "./KeyboardShortcutsHelp";
import { GuidedTour } from "./GuidedTour";
import { OnboardingDialog } from "./OnboardingDialog";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { WorkflowNav } from "./WorkflowNav";

interface Props {
  children: React.ReactNode;
}

export function Layout({ children }: Props) {
  const sidebarOpen = useLayoutStore((s) => s.sidebarOpen);
  const setSidebarOpen = useLayoutStore((s) => s.setSidebarOpen);
  useGlobalFileDrop();

  // Collapse sidebar on narrow viewports so content is visible on first paint (§G.7)
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 1023px)");
    const apply = () => {
      if (mq.matches) setSidebarOpen(false);
    };
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [setSidebarOpen]);

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      {/* Mobile backdrop */}
      {sidebarOpen && (
        <button
          type="button"
          aria-label="Close sidebar"
          className="fixed inset-0 z-20 bg-black/40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <Sidebar />

      <div className="flex flex-col flex-1 overflow-hidden min-w-0">
        <TopBar />
        <WorkflowNav />
        <main className="flex-1 overflow-auto bg-canvas-bg p-4 sm:p-5">
          <div className="max-w-[1920px] mx-auto w-full min-w-0">{children}</div>
        </main>
      </div>
      <OnboardingDialog />
      <GuidedTour />
      <CommandPalette />
      <KeyboardShortcutsHelp />
    </div>
  );
}
