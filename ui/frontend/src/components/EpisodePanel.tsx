import React, { useState } from 'react';
import { ChevronRightIcon, ChevronDownIcon } from './icons';

interface TrajectoryStep {
  observation: any;
  action: any;
  reward: number;
  done: boolean;
  chat_completions?: any;
  model_response?: string;
}

interface Trajectory {
  uid: string;
  name?: string;  // e.g., "solver", "judge"
  reward: number;
  steps: TrajectoryStep[];
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

interface EpisodePanelProps {
  episodes: Episode[];
  selectedStep: number | null;
  loading?: boolean;
}

/**
 * EpisodePanel - Displays episodes for a selected step with expandable trajectories.
 */
export const EpisodePanel: React.FC<EpisodePanelProps> = ({
  episodes,
  selectedStep,
  loading = false,
}) => {
  const [expandedEpisode, setExpandedEpisode] = useState<string | null>(null);
  const [expandedTrajectory, setExpandedTrajectory] = useState<string | null>(null);

  // Filter episodes by selected step
  const filteredEpisodes = selectedStep !== null
    ? episodes.filter((ep) => ep.step === selectedStep)
    : [];

  if (selectedStep === null) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500">
        <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mb-3">
          <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
          </svg>
        </div>
        <p className="text-sm font-medium">Click a point on the chart</p>
        <p className="text-xs text-gray-400 mt-1">to view episodes from that step</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="w-6 h-6 border-2 border-gray-200 border-t-black rounded-full animate-spin" />
      </div>
    );
  }

  if (filteredEpisodes.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500">
        <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mb-3">
          <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <p className="text-sm font-medium">No episodes at step {selectedStep}</p>
        <p className="text-xs text-gray-400 mt-1">This step may only contain metrics</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
        <h3 className="text-sm font-semibold text-black">
          Step {selectedStep}
        </h3>
        <p className="text-xs text-gray-500 mt-0.5">
          {filteredEpisodes.length} episode{filteredEpisodes.length !== 1 ? 's' : ''}
        </p>
      </div>

      {/* Episode List */}
      <div className="flex-1 overflow-y-auto">
        {filteredEpisodes.map((episode) => (
          <EpisodeCard
            key={episode.id}
            episode={episode}
            isExpanded={expandedEpisode === episode.id}
            onToggle={() => setExpandedEpisode(
              expandedEpisode === episode.id ? null : episode.id
            )}
            expandedTrajectory={expandedTrajectory}
            onTrajectoryToggle={(uid) => setExpandedTrajectory(
              expandedTrajectory === uid ? null : uid
            )}
          />
        ))}
      </div>
    </div>
  );
};

interface EpisodeCardProps {
  episode: Episode;
  isExpanded: boolean;
  onToggle: () => void;
  expandedTrajectory: string | null;
  onTrajectoryToggle: (uid: string) => void;
}

const EpisodeCard: React.FC<EpisodeCardProps> = ({
  episode,
  isExpanded,
  onToggle,
  expandedTrajectory,
  onTrajectoryToggle,
}) => {
  const trajectories = episode.data?.trajectories || [];

  return (
    <div className="border-b border-gray-200">
      {/* Episode Header */}
      <button
        onClick={onToggle}
        className="w-full px-4 py-3 flex items-center gap-3 hover:bg-gray-50 transition-colors text-left"
      >
        {isExpanded ? (
          <ChevronDownIcon sx={{ fontSize: 16 }} className="text-gray-400 flex-shrink-0" />
        ) : (
          <ChevronRightIcon sx={{ fontSize: 16 }} className="text-gray-400 flex-shrink-0" />
        )}
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={`
              inline-flex items-center px-2 py-0.5 rounded text-xs font-medium
              ${episode.is_correct 
                ? 'bg-green-100 text-green-800' 
                : 'bg-red-100 text-red-800'
              }
            `}>
              {episode.is_correct ? '✓ Correct' : '✗ Incorrect'}
            </span>
            <span className="text-xs text-gray-500">
              Reward: {episode.reward?.toFixed(3) ?? 'N/A'}
            </span>
          </div>
          <p className="text-xs text-gray-600 mt-1 truncate font-mono">
            {episode.id}
          </p>
        </div>

        <span className="text-xs text-gray-400 flex-shrink-0">
          {trajectories.length} traj{trajectories.length !== 1 ? 's' : ''}
        </span>
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="px-4 pb-4 bg-gray-50">
          {/* Task Info */}
          <div className="mb-3 p-3 bg-white rounded-lg border border-gray-200">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">
              Task
            </p>
            <p className="text-sm text-black">
              {getTaskSummary(episode.task)}
            </p>
          </div>

          {/* Trajectories */}
          {trajectories.map((trajectory, idx) => (
            <TrajectoryCard
              key={trajectory.uid || idx}
              trajectory={trajectory}
              index={idx}
              isExpanded={expandedTrajectory === (trajectory.uid || `${episode.id}-${idx}`)}
              onToggle={() => onTrajectoryToggle(trajectory.uid || `${episode.id}-${idx}`)}
            />
          ))}
        </div>
      )}
    </div>
  );
};

interface TrajectoryCardProps {
  trajectory: Trajectory;
  index: number;
  isExpanded: boolean;
  onToggle: () => void;
}

const TrajectoryCard: React.FC<TrajectoryCardProps> = ({
  trajectory,
  index,
  isExpanded,
  onToggle,
}) => {
  return (
    <div className="mt-2 bg-white rounded-lg border border-gray-200 overflow-hidden">
      {/* Trajectory Header */}
      <button
        onClick={onToggle}
        className="w-full px-3 py-2 flex items-center gap-2 hover:bg-gray-50 transition-colors text-left"
      >
        {isExpanded ? (
          <ChevronDownIcon sx={{ fontSize: 14 }} className="text-gray-400" />
        ) : (
          <ChevronRightIcon sx={{ fontSize: 14 }} className="text-gray-400" />
        )}
        <span className="text-xs font-medium text-gray-700 capitalize">
          {trajectory.name || `Trajectory ${index + 1}`}
        </span>
        <span className="text-xs text-gray-500">
          • {trajectory.steps?.length || 0} step{trajectory.steps?.length !== 1 ? 's' : ''}
        </span>
        <span className="text-xs text-gray-500 ml-auto">
          Reward: {trajectory.reward?.toFixed(3) ?? 'N/A'}
        </span>
      </button>

      {/* Trajectory Steps */}
      {isExpanded && trajectory.steps && (
        <div className="border-t border-gray-200">
          {trajectory.steps.map((step, stepIdx) => (
            <StepView key={stepIdx} step={step} index={stepIdx} />
          ))}
        </div>
      )}
    </div>
  );
};

interface StepViewProps {
  step: TrajectoryStep;
  index: number;
}

const StepView: React.FC<StepViewProps> = ({ step, index }) => {
  return (
    <div className={`px-3 py-3 ${index > 0 ? 'border-t border-gray-100' : ''}`}>
      {/* Step Header */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-gray-500">
          Step {index + 1}
        </span>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-gray-500">
            Reward: {step.reward?.toFixed(3) ?? '0'}
          </span>
          {step.done && (
            <span className="px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded text-xs">
              Done
            </span>
          )}
        </div>
      </div>

      {/* Observation / User Message */}
      <div className="mb-2">
        <p className="text-xs font-medium text-blue-600 mb-1">
          {step.observation != null ? 'Observation' : 'User Message'}
        </p>
        <div className="bg-blue-50 rounded p-2 text-xs text-gray-800 font-mono whitespace-pre-wrap break-words max-h-32 overflow-y-auto">
          {formatContent(
            step.observation != null 
              ? step.observation 
              : step.chat_completions?.[0]?.content
          )}
        </div>
      </div>

      {/* Model Response */}
      {(step.model_response || step.chat_completions?.[1]?.content) && (
        <div className="mb-2">
          <p className="text-xs font-medium text-purple-600 mb-1">Response</p>
          <div className="bg-purple-50 rounded p-2 text-xs text-gray-800 font-mono whitespace-pre-wrap break-words max-h-48 overflow-y-auto">
            {formatContent(step.model_response || step.chat_completions?.[1]?.content)}
          </div>
        </div>
      )}

      {/* Action */}
      <div>
        <p className="text-xs font-medium text-green-600 mb-1">Action</p>
        <div className="bg-green-50 rounded p-2 text-xs text-gray-800 font-mono whitespace-pre-wrap break-words max-h-32 overflow-y-auto">
          {formatContent(step.action)}
        </div>
      </div>
    </div>
  );
};

// Helper functions

function getTaskSummary(task: Record<string, any>): string {
  // Try common task fields
  if (task.question) return task.question;
  if (task.problem) return task.problem;
  if (task.prompt) return task.prompt;
  if (task.input) return task.input;
  if (task.text) return task.text;
  
  // Fallback to JSON
  const jsonStr = JSON.stringify(task);
  return jsonStr.length > 200 ? jsonStr.slice(0, 200) + '...' : jsonStr;
}

function formatContent(content: any): string {
  if (content === null || content === undefined) {
    return 'N/A';
  }
  if (typeof content === 'string') {
    return content;
  }
  try {
    return JSON.stringify(content, null, 2);
  } catch {
    return String(content);
  }
}

