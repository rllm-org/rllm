import React, { useState, useEffect } from 'react';

interface SourceMetadata {
  workflow_source?: string;
  workflow_class?: string;
  reward_fn_source?: string;
  reward_fn_name?: string;
  agent_source?: string;
  agent_class?: string;
}

interface Session {
  id: string;
  project: string;
  experiment: string;
  config: Record<string, any> | null;
  source_metadata?: SourceMetadata | null;
  created_at: string;
  completed_at: string | null;
}

interface WorkflowDiagramProps {
  session: Session | null;
}

type TabType = 'workflow' | 'reward' | 'agent';

interface TabConfig {
  id: TabType;
  label: string;
  hasContent: boolean;
  title?: string;
  source?: string;
  fallbackMessage?: string;
}

const TabButton: React.FC<{
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}> = ({ active, onClick, children }) => (
  <button
    className={`px-4 py-2 text-sm font-medium transition-colors ${
      active
        ? 'border-b-2 border-blue-500 text-blue-600 bg-white'
        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
    }`}
    onClick={onClick}
  >
    {children}
  </button>
);

const SourceCodePanel: React.FC<{
  title?: string;
  source?: string;
  fallbackMessage?: string;
}> = ({ title, source, fallbackMessage }) => (
  <div>
    {title && (
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
        <h3 className="text-sm font-semibold text-black">{title}</h3>
      </div>
    )}
    {source ? (
      <pre className="p-4 text-xs font-mono text-black bg-white overflow-x-auto">
        <code>{source}</code>
      </pre>
    ) : fallbackMessage ? (
      <div className="p-4 text-sm text-gray-600">
        <p>Built-in function: <code className="bg-gray-100 px-2 py-1 rounded">{title}</code></p>
        <p className="mt-2 text-gray-500">{fallbackMessage}</p>
      </div>
    ) : null}
  </div>
);

export const WorkflowDiagram: React.FC<WorkflowDiagramProps> = ({ session }) => {
  const [activeTab, setActiveTab] = useState<TabType>('workflow');

  const metadata = session?.source_metadata;

  // Build tab configuration
  const tabs: TabConfig[] = [
    {
      id: 'workflow',
      label: 'Workflow',
      hasContent: !!metadata?.workflow_source,
      title: metadata?.workflow_class,
      source: metadata?.workflow_source,
    },
    {
      id: 'reward',
      label: 'Reward Function',
      hasContent: !!metadata?.reward_fn_source || !!metadata?.reward_fn_name,
      title: metadata?.reward_fn_name,
      source: metadata?.reward_fn_source,
      fallbackMessage: 'Source code not available for built-in functions.',
    },
    {
      id: 'agent',
      label: 'Agent',
      hasContent: !!metadata?.agent_source,
      title: metadata?.agent_class,
      source: metadata?.agent_source,
    },
  ];

  const availableTabs = tabs.filter(t => t.hasContent);

  // Set default tab to first available
  useEffect(() => {
    if (availableTabs.length > 0 && !availableTabs.find(t => t.id === activeTab)) {
      setActiveTab(availableTabs[0].id);
    }
  }, [availableTabs, activeTab]);

  if (!metadata || availableTabs.length === 0) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-50 border border-gray-200 rounded-xl">
        <p className="text-sm text-gray-500">No source code available for this session</p>
      </div>
    );
  }

  const currentTab = tabs.find(t => t.id === activeTab);

  return (
    <div className="w-full bg-white border border-gray-200 rounded-xl overflow-hidden">
      {/* Tab bar */}
      <div className="flex border-b border-gray-200 bg-gray-50">
        {availableTabs.map(tab => (
          <TabButton
            key={tab.id}
            active={activeTab === tab.id}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </TabButton>
        ))}
      </div>

      {/* Content */}
      <div className="overflow-x-auto">
        {currentTab && (
          <SourceCodePanel
            title={currentTab.title}
            source={currentTab.source}
            fallbackMessage={currentTab.fallbackMessage}
          />
        )}
      </div>
    </div>
  );
};
