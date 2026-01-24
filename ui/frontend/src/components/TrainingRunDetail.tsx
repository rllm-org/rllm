import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useMetricsSSE } from '../hooks/useSSE';
import { RewardChart, getAvailableMetrics } from './RewardChart';
import { EpisodePanel } from './EpisodePanel';
import { WorkflowDiagram } from './WorkflowDiagram';
import { ProgressBar } from './ProgressBar';
import {
  TimelineIcon,
  WarningIcon,
  ChevronLeftIcon,
  InfoIcon,
  AccountTreeIcon,
} from './icons';

interface Session {
  id: string;
  project: string;
  experiment: string;
  config: Record<string, any> | null;
  source_metadata?: {
    workflow_source?: string;
    workflow_class?: string;
    reward_fn_source?: string;
    reward_fn_name?: string;
    agent_source?: string;
    agent_class?: string;
  } | null;
  created_at: string;
  completed_at: string | null;
}

interface Episode {
  id: string;
  session_id: string;
  step: number;
  task: Record<string, any>;
  is_correct: boolean;
  reward: number | null;
  data: {
    trajectories: Trajectory[];
  };
  created_at: string;
}

interface Trajectory {
  uid: string;
  reward: number;
  steps: TrajectoryStep[];
}

interface TrajectoryStep {
  observation: any;
  action: any;
  reward: number;
  done: boolean;
  chat_completions?: any;
  model_response?: any;
}

// Helper function to format config values for display
const formatConfigValue = (value: any): string => {
  if (value === null || value === undefined) return 'N/A';
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (Array.isArray(value)) return value.join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
};

export const TrainingRunDetail: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();

  const [session, setSession] = useState<Session | null>(null);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [activeTab, setActiveTab] = useState<'training' | 'logs' | 'metadata' | 'workflow'>('training');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<string>('reward/mean');

  // Enable metrics streaming for training tab
  const { metrics, isConnected } = useMetricsSSE({
    sessionId: sessionId || '',
    enabled: activeTab === 'training' && !!sessionId,
  });

  useEffect(() => {
    if (sessionId) {
      fetchSessionDetails();
      fetchEpisodes();
    }
  }, [sessionId]);

  // Poll for new episodes while connected (live training)
  useEffect(() => {
    if (!sessionId || !isConnected) return;
    
    const interval = setInterval(() => {
      fetchEpisodes();
    }, 3000); // Poll every 3 seconds
    
    return () => clearInterval(interval);
  }, [sessionId, isConnected]);

  const fetchSessionDetails = async () => {
    try {
      const response = await fetch(`http://localhost:3000/api/sessions/${sessionId}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setSession(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchEpisodes = async () => {
    try {
      const response = await fetch(`http://localhost:3000/api/episodes?session_id=${sessionId}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setEpisodes(data);
    } catch (err: any) {
      console.error('Error fetching episodes:', err);
    }
  };

  const handleStepClick = (step: number) => {
    setSelectedStep(step);
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-4 border-gray-200 border-t-black rounded-full animate-spin" />
          <p className="text-sm text-gray-500">Loading training run details...</p>
        </div>
      </div>
    );
  }

  if (error || !session) {
    return (
      <div className="flex-1 p-8">
        <div className="max-w-2xl mx-auto">
          <div className="bg-gray-50 border border-gray-300 rounded-xl p-6">
            <div className="flex items-start gap-3">
              <WarningIcon sx={{ fontSize: 28 }} className="text-black" />
              <div className="flex-1">
                <h3 className="text-sm font-semibold text-black mb-1">Error</h3>
                <p className="text-sm text-gray-700">{error || 'Session not found'}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Extract progress metrics for progress bar
  const latestMetrics = metrics.length > 0 ? metrics[metrics.length - 1]?.data : null;
  const doneFrac = latestMetrics?.['progress/done_frac'] || 0;
  const currentBatch = latestMetrics?.['progress/batch'] || 0;
  const currentEpoch = latestMetrics?.['progress/epoch'] || 0;
  const totalEpochs = session?.config?.trainer?.total_epochs || 0;
  const totalBatches = doneFrac > 0 ? Math.round((currentBatch + 1) / doneFrac) : null;

  return (
    <div className="flex-1 p-8 overflow-auto">
      <div className="w-full">
        {/* Session header */}
        <div className="mb-8">
          <h1 className="text-xl font-semibold text-black mb-2">
            {session.project} / {session.experiment}
          </h1>
          <code className="text-sm text-gray-600">
            Session ID: {session.id}
          </code>
          {/* Core metadata from rllm_config */}
          <div className="flex flex-wrap gap-4 mt-3">
            {session.config?.algorithm?.adv_estimator && (
              <div className="flex items-center gap-1.5 text-sm">
                <span className="text-gray-500">Algorithm:</span>
                <span className="text-black font-medium">
                  {session.config.algorithm.adv_estimator}
                </span>
              </div>
            )}
            {session.config?.backend && (
              <div className="flex items-center gap-1.5 text-sm">
                <span className="text-gray-500">Backend:</span>
                <span className="text-black font-medium">
                  {session.config.backend}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200 mb-6">
          <div className="flex items-center justify-between">
            <nav className="flex gap-6">
              <button
                onClick={() => setActiveTab('training')}
                className={`
                  inline-flex items-center gap-2 px-1 py-3 border-b-2 text-sm font-medium transition-colors duration-200
                  ${activeTab === 'training'
                    ? 'border-black text-black'
                    : 'border-transparent text-gray-600 hover:text-black hover:border-gray-300'
                  }
                `}
              >
                <TimelineIcon sx={{ fontSize: 20 }} />
                Training Progress
              </button>
              <button
                onClick={() => setActiveTab('workflow')}
                className={`
                  inline-flex items-center gap-2 px-1 py-3 border-b-2 text-sm font-medium transition-colors duration-200
                  ${activeTab === 'workflow'
                    ? 'border-black text-black'
                    : 'border-transparent text-gray-600 hover:text-black hover:border-gray-300'
                  }
                `}
              >
                <AccountTreeIcon sx={{ fontSize: 20 }} />
                Workflow
              </button>
              <button
                onClick={() => setActiveTab('metadata')}
                className={`
                  inline-flex items-center gap-2 px-1 py-3 border-b-2 text-sm font-medium transition-colors duration-200
                  ${activeTab === 'metadata'
                    ? 'border-black text-black'
                    : 'border-transparent text-gray-600 hover:text-black hover:border-gray-300'
                  }
                `}
              >
                <InfoIcon sx={{ fontSize: 20 }} />
                Metadata
              </button>
            </nav>

            {/* Progress Bar - Top Right */}
            {activeTab === 'training' && metrics.length > 0 && totalEpochs > 0 && (
              <ProgressBar
                doneFrac={doneFrac}
                currentEpoch={currentEpoch}
                totalEpochs={totalEpochs}
                currentBatch={currentBatch}
                totalBatches={totalBatches}
              />
            )}
          </div>
        </div>

        {/* Tab content */}
        {activeTab === 'training' ? (
          <div className="w-full">
            <div className="flex flex-col lg:flex-row gap-6">
            {/* Chart Panel */}
            <div className="w-full lg:w-1/2">
              <div className="mb-4">
                <select
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                  className="text-sm font-medium border border-gray-300 rounded-lg px-3 py-2 bg-white text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  {getAvailableMetrics(metrics).map((metric) => (
                    <option key={metric} value={metric}>
                      {metric}
                    </option>
                  ))}
                </select>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-4">
                <div className="h-[400px] outline-none focus:outline-none [&_*]:outline-none">
                  <RewardChart
                    metrics={metrics}
                    selectedMetric={selectedMetric}
                    selectedStep={selectedStep}
                    onStepClick={handleStepClick}
                  />
                </div>
              </div>
            </div>

            {/* Episodes Panel */}
            <div className="w-full lg:w-1/2">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-semibold text-black">
                  Episodes
                </h2>
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-black border border-gray-300">
                  {episodes.length} total
                </span>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden h-[450px]">
                <EpisodePanel
                  episodes={episodes}
                  selectedStep={selectedStep}
                />
              </div>
            </div>
            </div>
          </div>
        ) : activeTab === 'metadata' ? (
          <div className="space-y-6">
            {session.config ? (
              <>
                {/* Dynamic config sections - renders all top-level keys */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {Object.entries(session.config).map(([sectionKey, sectionValue]) => {
                    // Skip non-object sections (they'll be shown in raw config)
                    if (typeof sectionValue !== 'object' || sectionValue === null) {
                      return null;
                    }

                    return (
                      <div key={sectionKey} className="bg-white border border-gray-200 rounded-xl p-6">
                        <h3 className="text-lg font-semibold text-black mb-4 capitalize">
                          {sectionKey.replace(/_/g, ' ')}
                        </h3>
                        <dl className="space-y-3">
                          {Object.entries(sectionValue).map(([key, value]) => (
                            <div key={key} className="flex justify-between items-start">
                              <dt className="text-sm font-medium text-gray-600">{key}</dt>
                              <dd className="text-sm text-black">
                                {formatConfigValue(value)}
                              </dd>
                            </div>
                          ))}
                        </dl>
                      </div>
                    );
                  })}
                </div>

                {/* Raw Config - Collapsible */}
                <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                  <details className="group">
                    <summary className="px-6 py-4 cursor-pointer hover:bg-gray-50 transition-colors flex items-center justify-between">
                      <span className="text-sm font-medium text-black">Raw Configuration (JSON)</span>
                      <ChevronLeftIcon
                        sx={{ fontSize: 20 }}
                        className="transform transition-transform group-open:rotate-90"
                      />
                    </summary>
                    <div className="border-t border-gray-200">
                      <pre className="bg-gray-900 text-gray-300 p-5 text-xs overflow-auto max-h-96">
                        {JSON.stringify(session.config, null, 2)}
                      </pre>
                    </div>
                  </details>
                </div>
              </>
            ) : (
              <div className="bg-white border border-gray-200 rounded-xl p-12 text-center">
                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <InfoIcon sx={{ fontSize: 32 }} className="text-gray-500" />
                </div>
                <p className="text-sm text-gray-500">No configuration data available</p>
              </div>
            )}
          </div>
        ) : activeTab === 'workflow' ? (
          <div className="w-full">
            <div className="mb-4">
              <h2 className="text-lg font-semibold text-black">Source Code</h2>
            </div>
            <WorkflowDiagram session={session} />
          </div>
        ) : null}
      </div>
    </div>
  );
};
