import { useLayoutStore } from "../../store/layout";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { WorkflowNav } from "./WorkflowNav";

interface Props {
  children: React.ReactNode;
}

export function Layout({ children }: Props) {
  const sidebarOpen = useLayoutStore((s) => s.sidebarOpen);
  const setSidebarOpen = useLayoutStore((s) => s.setSidebarOpen);

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
    </div>
  );
}
