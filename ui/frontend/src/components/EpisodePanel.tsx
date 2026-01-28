import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { ChevronRightIcon, ChevronDownIcon, SearchIcon } from './icons';
import { HighlightedText, textContains } from './HighlightedText';

interface TrajectoryStep {
  observation?: any;
  thought?: string;
  action?: any;
  model_response?: string;
  chat_completions?: any[];
  info?: any;
  reward: number;
  done: boolean;
  // Internal/training fields (not displayed)
  mc_return?: number;
  advantage?: number;
  prompt_ids?: any;
  response_ids?: any;
  logprobs?: any;
  // Allow any additional fields
  [key: string]: any;
}

interface Trajectory {
  uid: string;
  name?: string;
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

// Represents a match location within the episode data
interface MatchLocation {
  episodeId: string;
  trajectoryUid: string;
  stepIndex: number;
  field: string; // Now supports any field name dynamically
}

interface EpisodePanelProps {
  episodes: Episode[];
  selectedStep: number | null;
  sessionId?: string;
  loading?: boolean;
}

// API search response type
interface SearchResponse {
  episodes: (Episode & { rank?: number })[];
  matched_terms: string[];
}

// API search function
const searchEpisodesAPI = async (
  query: string,
  sessionId: string,
  limit: number = 50,
  step?: number | null
): Promise<SearchResponse> => {
  const params = new URLSearchParams({
    q: query,
    session_id: sessionId,
    limit: String(limit),
  });
  if (step !== null && step !== undefined) {
    params.set('step', String(step));
  }
  const response = await fetch(`http://localhost:3000/api/episodes/search?${params}`);
  if (!response.ok) {
    throw new Error(`Search failed: ${response.status}`);
  }
  return response.json();
};

/**
 * EpisodePanel - Displays episodes for a selected step with expandable trajectories and inline search.
 */
export const EpisodePanel: React.FC<EpisodePanelProps> = ({
  episodes,
  selectedStep,
  sessionId,
  loading = false,
}) => {
  const [expandedEpisodes, setExpandedEpisodes] = useState<Set<string>>(new Set());
  const [expandedTrajectories, setExpandedTrajectories] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [matchLocations, setMatchLocations] = useState<MatchLocation[]>([]);
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0);

  // API search state
  const [searchResults, setSearchResults] = useState<(Episode & { rank?: number })[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  // Matched terms from API (stemmed terms for PostgreSQL, original terms for SQLite)
  const [matchedTerms, setMatchedTerms] = useState<string[]>([]);

  const currentMatchRef = useRef<HTMLSpanElement>(null);
  const prevDebouncedQuery = useRef('');
  const shouldScrollRef = useRef(false);

  // Derive sessionId from props or first episode
  const effectiveSessionId = sessionId || episodes[0]?.session_id;

  // Debounce search query (300ms)
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchQuery);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Memoize filtered episodes - API handles step filtering when searching
  const filteredEpisodes = useMemo(() => {
    if (debouncedQuery.trim()) {
      // API already filters by step, just return results
      return searchResults;
    }
    // When not searching, filter by selected step
    return selectedStep !== null
      ? episodes.filter((ep) => ep.step === selectedStep)
      : [];
  }, [episodes, selectedStep, debouncedQuery, searchResults]);

  // Perform API search when debounced query changes
  useEffect(() => {
    if (!debouncedQuery.trim()) {
      setSearchResults([]);
      setMatchLocations([]);
      setCurrentMatchIndex(0);
      setSearchError(null);
      setMatchedTerms([]);
      return;
    }

    if (!effectiveSessionId) {
      setSearchError('No session ID available');
      return;
    }

    const performSearch = async () => {
      setSearchLoading(true);
      setSearchError(null);

      try {
        const response = await searchEpisodesAPI(debouncedQuery, effectiveSessionId, 50, selectedStep);
        const { episodes: results = [], matched_terms: terms = [] } = response;
        setSearchResults(results);
        setMatchedTerms(terms);

        // Build match locations from results (for highlighting)
        // API already filters by step, so results are pre-filtered
        // Use matched_terms for matching to support PostgreSQL stemming
        const matches: MatchLocation[] = [];
        const expandEpisodes = new Set<string>();
        const expandTrajectories = new Set<string>();

        results.forEach((episode) => {
          // Check task for highlighting
          const taskText = getTaskSummary(episode.task);
          if (textContains(taskText, debouncedQuery, terms)) {
            matches.push({
              episodeId: episode.id,
              trajectoryUid: '',
              stepIndex: -1,
              field: 'task',
            });
            expandEpisodes.add(episode.id);
          }

          // Check trajectory steps for highlighting
          const trajectories = episode.data?.trajectories || [];
          trajectories.forEach((trajectory, trajIdx) => {
            const trajUid = trajectory.uid || `${episode.id}-${trajIdx}`;

            trajectory.steps?.forEach((step, stepIdx) => {
              const visibleFields = getVisibleFields(step);

              visibleFields.forEach(({ key, value }) => {
                const fieldText = formatFieldValue(value);
                if (textContains(fieldText, debouncedQuery, terms)) {
                  matches.push({
                    episodeId: episode.id,
                    trajectoryUid: trajUid,
                    stepIndex: stepIdx,
                    field: key
                  });
                  expandEpisodes.add(episode.id);
                  expandTrajectories.add(trajUid);
                }
              });
            });
          });
        });

        setMatchLocations(matches);

        // Reset index when query changes
        if (prevDebouncedQuery.current !== debouncedQuery) {
          setCurrentMatchIndex(0);
          prevDebouncedQuery.current = debouncedQuery;
          if (matches.length > 0) {
            shouldScrollRef.current = true;
          }
        }

        // Auto-expand matching episodes and trajectories
        if (matches.length > 0) {
          setExpandedEpisodes(expandEpisodes);
          setExpandedTrajectories(expandTrajectories);
        }
      } catch (err) {
        setSearchError(err instanceof Error ? err.message : 'Search failed');
        setSearchResults([]);
        setMatchLocations([]);
        setMatchedTerms([]);
      } finally {
        setSearchLoading(false);
      }
    };

    performSearch();
  }, [debouncedQuery, effectiveSessionId, selectedStep]);

  // Scroll to current match ONLY when explicitly navigating
  useEffect(() => {
    if (shouldScrollRef.current && currentMatchRef.current) {
      currentMatchRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
      shouldScrollRef.current = false;
    }
  }, [currentMatchIndex]);

  const navigateToNextMatch = useCallback(() => {
    if (matchLocations.length === 0) return;
    shouldScrollRef.current = true;
    setCurrentMatchIndex((prev) => (prev + 1) % matchLocations.length);
  }, [matchLocations.length]);

  const navigateToPrevMatch = useCallback(() => {
    if (matchLocations.length === 0) return;
    shouldScrollRef.current = true;
    setCurrentMatchIndex((prev) => (prev - 1 + matchLocations.length) % matchLocations.length);
  }, [matchLocations.length]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      if (e.shiftKey) {
        navigateToPrevMatch();
      } else {
        navigateToNextMatch();
      }
    }
  };

  const toggleEpisode = (episodeId: string) => {
    setExpandedEpisodes((prev) => {
      const next = new Set(prev);
      if (next.has(episodeId)) {
        next.delete(episodeId);
      } else {
        next.add(episodeId);
      }
      return next;
    });
  };

  const toggleTrajectory = (trajUid: string) => {
    setExpandedTrajectories((prev) => {
      const next = new Set(prev);
      if (next.has(trajUid)) {
        next.delete(trajUid);
      } else {
        next.add(trajUid);
      }
      return next;
    });
  };

  // Get current match for highlighting
  const currentMatch = matchLocations[currentMatchIndex] || null;

  // Determine which episodes to display
  const displayEpisodes = debouncedQuery.trim()
    ? filteredEpisodes  // All matching episodes when searching
    : filteredEpisodes; // Filtered by step when not searching

  // Render the search bar (always visible)
  const renderSearchBar = () => (
    <div className="px-4 py-2 border-b border-gray-200 bg-white">
      <div className="flex items-center gap-2">
        <div className="flex-1 relative">
          <input
            type="text"
            placeholder={selectedStep !== null
              ? `Search step ${selectedStep}...`
              : "Search all episodes..."
            }
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            className="w-full pl-8 pr-3 py-1.5 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-black focus:border-black"
          />
          <SearchIcon
            sx={{ fontSize: 16 }}
            className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-400"
          />
        </div>
        {matchLocations.length > 0 && (
          <>
            <button
              onClick={navigateToPrevMatch}
              className="p-1 hover:bg-gray-100 rounded text-gray-600"
              title="Previous match (Shift+Enter)"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
              </svg>
            </button>
            <button
              onClick={navigateToNextMatch}
              className="p-1 hover:bg-gray-100 rounded text-gray-600"
              title="Next match (Enter)"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            <span className="text-xs text-gray-500 min-w-[3rem] text-center">
              {currentMatchIndex + 1}/{matchLocations.length}
            </span>
          </>
        )}
      </div>
    </div>
  );

  // Main render
  return (
    <div className="h-full flex flex-col">
      {/* Search Bar - Always visible at the top */}
      {renderSearchBar()}

      {/* Content area */}
      {loading || searchLoading ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="flex flex-col items-center gap-2">
            <div className="w-6 h-6 border-2 border-gray-200 border-t-black rounded-full animate-spin" />
            {searchLoading && <span className="text-xs text-gray-500">Searching...</span>}
          </div>
        </div>
      ) : searchError ? (
        // Search error
        <div className="flex-1 flex flex-col items-center justify-center text-gray-500">
          <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center mb-3">
            <svg className="w-6 h-6 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <p className="text-sm font-medium text-red-600">Search failed</p>
          <p className="text-xs text-gray-400 mt-1">{searchError}</p>
        </div>
      ) : displayEpisodes.length === 0 && !debouncedQuery.trim() ? (
        // No episodes and no search
        selectedStep === null ? (
          <div className="flex-1 flex flex-col items-center justify-center text-gray-500">
            <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mb-3">
              <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
              </svg>
            </div>
            <p className="text-sm font-medium">Click a point on the chart</p>
            <p className="text-xs text-gray-400 mt-1">to view episodes from that step</p>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-gray-500">
            <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mb-3">
              <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <p className="text-sm font-medium">No episodes at step {selectedStep}</p>
            <p className="text-xs text-gray-400 mt-1">This step may only contain metrics</p>
          </div>
        )
      ) : displayEpisodes.length === 0 && debouncedQuery.trim() ? (
        // Searching but no results
        <div className="flex-1 flex flex-col items-center justify-center text-gray-500">
          <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mb-3">
            <SearchIcon sx={{ fontSize: 24 }} className="text-gray-400" />
          </div>
          <p className="text-sm font-medium">No matches found</p>
          <p className="text-xs text-gray-400 mt-1">Try a different search term</p>
        </div>
      ) : (
        <>
          {/* Step Header - only show when viewing a specific step */}
          {selectedStep !== null && !debouncedQuery.trim() && (
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
              <h3 className="text-sm font-semibold text-black">
                Step {selectedStep}
              </h3>
              <p className="text-xs text-gray-500 mt-0.5">
                {displayEpisodes.length} episode{displayEpisodes.length !== 1 ? 's' : ''}
              </p>
            </div>
          )}


          {/* Episode List */}
          <div className="flex-1 overflow-y-auto">
            {displayEpisodes.map((episode) => (
              <EpisodeCard
                key={episode.id}
                episode={episode}
                isExpanded={expandedEpisodes.has(episode.id)}
                onToggle={() => toggleEpisode(episode.id)}
                expandedTrajectories={expandedTrajectories}
                onTrajectoryToggle={toggleTrajectory}
                searchQuery={debouncedQuery}
                searchTerms={matchedTerms}
                currentMatch={currentMatch}
                currentMatchRef={currentMatchRef}
              />
            ))}
          </div>
        </>
      )}
    </div>
  );
};

interface EpisodeCardProps {
  episode: Episode;
  isExpanded: boolean;
  onToggle: () => void;
  expandedTrajectories: Set<string>;
  onTrajectoryToggle: (uid: string) => void;
  searchQuery: string;
  searchTerms: string[];
  currentMatch: MatchLocation | null;
  currentMatchRef: React.RefObject<HTMLSpanElement>;
}

const EpisodeCard: React.FC<EpisodeCardProps> = ({
  episode,
  isExpanded,
  onToggle,
  expandedTrajectories,
  onTrajectoryToggle,
  searchQuery,
  searchTerms,
  currentMatch,
  currentMatchRef,
}) => {
  const trajectories = episode.data?.trajectories || [];
  const isCurrentMatchInTask = currentMatch?.episodeId === episode.id && currentMatch?.field === 'task';

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
              Step {episode.step} • Reward: {episode.reward?.toFixed(3) ?? 'N/A'}
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
              <HighlightedText
                text={getTaskSummary(episode.task)}
                searchQuery={searchQuery}
                searchTerms={searchTerms}
                isCurrentMatch={isCurrentMatchInTask}
                matchRef={isCurrentMatchInTask ? currentMatchRef : undefined}
              />
            </p>
          </div>

          {/* Trajectories */}
          {trajectories.map((trajectory, idx) => {
            const trajUid = trajectory.uid || `${episode.id}-${idx}`;
            return (
              <TrajectoryCard
                key={trajUid}
                trajectory={trajectory}
                index={idx}
                episodeId={episode.id}
                trajUid={trajUid}
                isExpanded={expandedTrajectories.has(trajUid)}
                onToggle={() => onTrajectoryToggle(trajUid)}
                searchQuery={searchQuery}
                searchTerms={searchTerms}
                currentMatch={currentMatch}
                currentMatchRef={currentMatchRef}
              />
            );
          })}
        </div>
      )}
    </div>
  );
};

interface TrajectoryCardProps {
  trajectory: Trajectory;
  index: number;
  episodeId: string;
  trajUid: string;
  isExpanded: boolean;
  onToggle: () => void;
  searchQuery: string;
  searchTerms: string[];
  currentMatch: MatchLocation | null;
  currentMatchRef: React.RefObject<HTMLSpanElement>;
}

const TrajectoryCard: React.FC<TrajectoryCardProps> = ({
  trajectory,
  index,
  episodeId,
  trajUid,
  isExpanded,
  onToggle,
  searchQuery,
  searchTerms,
  currentMatch,
  currentMatchRef,
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
            <StepView
              key={stepIdx}
              step={step}
              index={stepIdx}
              episodeId={episodeId}
              trajUid={trajUid}
              searchQuery={searchQuery}
              searchTerms={searchTerms}
              currentMatch={currentMatch}
              currentMatchRef={currentMatchRef}
            />
          ))}
        </div>
      )}
    </div>
  );
};

interface StepViewProps {
  step: TrajectoryStep;
  index: number;
  episodeId: string;
  trajUid: string;
  searchQuery: string;
  searchTerms: string[];
  currentMatch: MatchLocation | null;
  currentMatchRef: React.RefObject<HTMLSpanElement>;
}

const StepView: React.FC<StepViewProps> = ({
  step,
  index,
  episodeId,
  trajUid,
  searchQuery,
  searchTerms,
  currentMatch,
  currentMatchRef,
}) => {
  // Get all visible fields dynamically
  const visibleFields = getVisibleFields(step);

  const isCurrentMatch = (fieldKey: string) =>
    currentMatch?.episodeId === episodeId &&
    currentMatch?.trajectoryUid === trajUid &&
    currentMatch?.stepIndex === index &&
    currentMatch?.field === fieldKey;

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

      {/* Render all visible fields dynamically */}
      {visibleFields.map(({ key, value }, fieldIdx) => {
        const fieldConfig = getFieldConfig(key, value);
        const fieldContent = formatFieldValue(value);
        const isCurrent = isCurrentMatch(key);

        return (
          <div key={key} className={fieldIdx < visibleFields.length - 1 ? 'mb-2' : ''}>
            <p className={`text-xs font-medium ${fieldConfig.labelColor} mb-1`}>
              {fieldConfig.label}
            </p>
            <div className={`${fieldConfig.bgColor} rounded p-2 text-xs text-gray-800 font-mono whitespace-pre-wrap break-words ${fieldConfig.maxHeight} overflow-y-auto`}>
              <HighlightedText
                text={fieldContent}
                searchQuery={searchQuery}
                searchTerms={searchTerms}
                isCurrentMatch={isCurrent}
                matchRef={isCurrent ? currentMatchRef : undefined}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
};

// Helper functions

/**
 * Fields that should not be displayed to users (internal/training fields)
 */
const HIDDEN_FIELDS = new Set([
  'prompt_ids',
  'response_ids',
  'logprobs',
  'mc_return',
  'advantage',
  'reward', // Shown in header
  'done',   // Shown in header
]);

/**
 * Gets all visible fields from a step object in display order
 */
function getVisibleFields(step: TrajectoryStep): Array<{ key: string; value: any }> {
  const fields: Array<{ key: string; value: any }> = [];

  // Define display order for known fields
  const fieldOrder = [
    'observation',
    'thought',
    'model_response',
    'action',
    'chat_completions',
    'info',
  ];

  // First, add fields in preferred order if they exist
  fieldOrder.forEach(key => {
    if (step[key] != null && !HIDDEN_FIELDS.has(key)) {
      fields.push({ key, value: step[key] });
    }
  });

  // Then add any remaining fields not in the order list
  Object.keys(step).forEach(key => {
    if (
      step[key] != null &&
      !HIDDEN_FIELDS.has(key) &&
      !fieldOrder.includes(key)
    ) {
      fields.push({ key, value: step[key] });
    }
  });

  return fields;
}

/**
 * Gets the display configuration for a field (label, colors, etc.)
 */
function getFieldConfig(key: string, _value: any): {
  label: string;
  labelColor: string;
  bgColor: string;
  maxHeight: string;
} {
  // Handle special cases first
  if (key === 'observation') {
    return {
      label: 'Observation',
      labelColor: 'text-blue-600',
      bgColor: 'bg-blue-50',
      maxHeight: 'max-h-32',
    };
  }

  if (key === 'thought') {
    return {
      label: 'Thought',
      labelColor: 'text-amber-600',
      bgColor: 'bg-amber-50',
      maxHeight: 'max-h-32',
    };
  }

  if (key === 'model_response') {
    return {
      label: 'Response',
      labelColor: 'text-purple-600',
      bgColor: 'bg-purple-50',
      maxHeight: 'max-h-48',
    };
  }

  if (key === 'action') {
    return {
      label: 'Action',
      labelColor: 'text-green-600',
      bgColor: 'bg-green-50',
      maxHeight: 'max-h-32',
    };
  }

  if (key === 'chat_completions') {
    return {
      label: 'Chat Completions',
      labelColor: 'text-indigo-600',
      bgColor: 'bg-indigo-50',
      maxHeight: 'max-h-48',
    };
  }

  if (key === 'info') {
    return {
      label: 'Info',
      labelColor: 'text-gray-600',
      bgColor: 'bg-gray-50',
      maxHeight: 'max-h-32',
    };
  }

  // Default for unknown fields
  return {
    label: formatFieldLabel(key),
    labelColor: 'text-gray-600',
    bgColor: 'bg-gray-50',
    maxHeight: 'max-h-32',
  };
}

/**
 * Formats a field key into a readable label
 */
function formatFieldLabel(key: string): string {
  return key
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Formats a field value for display, with special handling for complex types
 */
function formatFieldValue(value: any): string {
  if (value === null || value === undefined) {
    return 'N/A';
  }

  // Handle chat_completions array specially
  if (Array.isArray(value) && value.length > 0 && value[0]?.role) {
    return value
      .map((msg: any) => {
        const role = msg.role || 'unknown';
        const content = msg.content || '';
        return `[${role.toUpperCase()}]: ${content}`;
      })
      .join('\n\n');
  }

  // Handle info object with tool_calls
  if (typeof value === 'object' && value.tool_calls) {
    const parts: string[] = [];

    if (value.tool_calls && Array.isArray(value.tool_calls)) {
      parts.push('Tool Calls:');
      value.tool_calls.forEach((call: any, idx: number) => {
        parts.push(`\n${idx + 1}. ${call.function?.name || 'unknown'}`);
        if (call.function?.arguments) {
          parts.push(`   Args: ${call.function.arguments}`);
        }
      });
    }

    // Add other info fields
    const otherKeys = Object.keys(value).filter(k => k !== 'tool_calls');
    if (otherKeys.length > 0) {
      parts.push('\n\nOther:');
      otherKeys.forEach(k => {
        parts.push(`${k}: ${formatContent(value[k])}`);
      });
    }

    return parts.join('\n');
  }

  // Default formatting
  return formatContent(value);
}

function getTaskSummary(task: Record<string, any>): string {
  if (task.question) return task.question;
  if (task.problem) return task.problem;
  if (task.prompt) return task.prompt;
  if (task.input) return task.input;
  if (task.text) return task.text;

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
