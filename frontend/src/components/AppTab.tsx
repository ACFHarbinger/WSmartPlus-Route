// src/components/AppTab.tsx
import React from 'react';

interface AppTabProps {
  children: React.ReactNode;
  className?: string;
}

export const AppTab: React.FC<AppTabProps> = ({ children, className = '' }) => (
  <div className={`p-6 space-y-4 h-full overflow-y-auto ${className}`}>
    {children}
  </div>
);

interface InternalTabButtonProps {
  label: string;
  isActive: boolean;
  onClick: () => void;
}

export const InternalTabButton: React.FC<InternalTabButtonProps> = ({
  label,
  isActive,
  onClick,
}) => (
  <div
    role="tab"
    aria-selected={isActive}
    tabIndex={0}
    onClick={onClick}
    onKeyDown={(e) => e.key === 'Enter' && onClick()}
    className={`px-6 py-3 font-medium transition cursor-pointer select-none
      ${isActive
        ? 'border-b-2 border-blue-500 text-blue-400'
        : 'text-gray-400 hover:text-gray-200'
      }`}
  >
    {label}
  </div>
);

interface AppContainerProps {
  tabs: { name: string; content: React.ReactNode }[];
}

export const AppContainer: React.FC<AppContainerProps> = ({ tabs }) => {
  const [activeTab, setActiveTab] = React.useState(tabs[0]?.name || '');

  return (
    <div className="flex flex-col h-full bg-gray-900 text-gray-200">
      {/* Tab Bar */}
      <nav className="flex-shrink-0 flex flex-wrap space-x-1 border-b border-gray-700 px-4 bg-gray-950">
        {tabs.map((tab) => (
          <InternalTabButton
            key={tab.name}
            label={tab.name}
            isActive={activeTab === tab.name}
            onClick={() => setActiveTab(tab.name)}
          />
        ))}
      </nav>

      {/* Tab Content */}
      <main className="flex-grow bg-gray-800 overflow-hidden">
        <div className="h-full">
          {tabs.find((tab) => tab.name === activeTab)?.content}
        </div>
      </main>
    </div>
  );
};