import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

interface Props {
  children: React.ReactNode;
}

export function Layout({ children }: Props) {
  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <Sidebar />
      <div className="flex flex-col flex-1 overflow-hidden">
        <TopBar />
        <main className="flex-1 overflow-auto bg-canvas-bg p-4 sm:p-5">
          <div className="max-w-[1920px] mx-auto w-full min-w-0">{children}</div>
        </main>
      </div>
    </div>
  );
}
