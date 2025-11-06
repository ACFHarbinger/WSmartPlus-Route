// src/App.tsx
import React, { useState } from 'react';
import {
  Settings,
  FilePlus,
  BrainCircuit,
  ClipboardCheck,
  Package,
} from 'lucide-react';
import { TestSimulatorApp } from './apps/TestSimulator';
import { GenerateDataApp } from './apps/GenerateData';
import { FileSystemToolsApp } from './apps/FileSystemTools';
import { RLApp } from './apps/ReinforcementLearning';
import { EvaluationApp } from './apps/Evaluation';

type AppName = 'Test Simulator' | 'Generate Data' | 'File System Tools' | 'Reinforcement Learning' | 'Evaluation';

const appConfig = {
  'Test Simulator': {
    icon: Settings,
    component: <TestSimulatorApp />,
  },
  'Generate Data': {
    icon: FilePlus,
    component: <GenerateDataApp />,
  },
  'File System Tools': {
    icon: Package,
    component: <FileSystemToolsApp />,
  },
  'Reinforcement Learning': {
    icon: BrainCircuit,
    component: <RLApp />,
  },
  'Evaluation': {
    icon: ClipboardCheck,
    component: <EvaluationApp />,
  },
};

const appNames = Object.keys(appConfig) as AppName[];

interface SidebarButtonProps {
  icon: React.ElementType;
  label: string;
  isActive: boolean;
  onClick: () => void;
}

const SidebarButton: React.FC<SidebarButtonProps> = ({ icon: Icon, label, isActive, onClick }) => (
  <button
    type="button"
    onClick={onClick}
    className={`flex items-center space-x-3 w-full px-3 py-3 rounded-lg text-left font-medium transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
      isActive
        ? 'bg-blue-600 text-white'
        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
    }`}
  >
    <Icon className="w-5 h-5 flex-shrink-0" />
    <span className="text-sm">{label}</span>
  </button>
);

const App: React.FC = () => {
  const [currentApp, setCurrentApp] = useState<AppName>(appNames[0]);

  return (
    <div className="flex h-screen bg-gray-900 text-gray-200 font-inter">
      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 bg-gray-800 p-4 space-y-2 flex flex-col shadow-lg">
        <h1 className="text-xl font-bold text-white mb-4 px-2">Unified GUI</h1>
        <nav className="flex-grow">
          {appNames.map(name => {
            const config = appConfig[name as AppName];
            return (
              <SidebarButton
                key={name}
                label={name}
                icon={config.icon}
                isActive={currentApp === name}
                onClick={() => setCurrentApp(name as AppName)}
              />
            );
          })}
        </nav>
        <footer className="flex-shrink-0 p-2 text-center text-gray-500 text-xs">
          React + TypeScript GUI
        </footer>
      </aside>

      {/* Main Content Area */}
      <div className="flex-grow flex flex-col overflow-hidden">
        <header className="flex-shrink-0 bg-gray-800 p-4 shadow-md z-10">
          <h2 className="text-2xl font-bold text-white">
            {currentApp}
          </h2>
        </header>
        <main className="flex-grow overflow-auto bg-gray-900">
          {appConfig[currentApp as AppName].component}
        </main>
      </div>
    </div>
  );
};

export default App;