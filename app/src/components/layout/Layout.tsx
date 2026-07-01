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
        <main className="flex-1 overflow-auto bg-canvas-bg p-5">
          {children}
        </main>
      </div>
    </div>
  );
}
