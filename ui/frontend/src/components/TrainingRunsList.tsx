import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { SearchIcon, WarningIcon, BarChartIcon } from './icons';

interface Session {
  id: string;
  project: string;
  experiment: string;
  config: Record<string, any> | null;
  created_at: string;
  completed_at: string | null;
}

export const TrainingRunsList: React.FC = () => {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:3000/api/sessions');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setSessions(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getBaseModel = (session: Session): string => {
    if (!session.config) return 'N/A';

    // Try common config paths for base model
    const model = session.config.model || session.config.base_model || session.config.model_name;

    // Ensure we return a string, not an object
    if (typeof model === 'string') return model;
    if (typeof model === 'object' && model !== null) {
      // If it's an object, try to extract a name property
      if ('name' in model && typeof model.name === 'string') return model.name;
      return 'Custom Model';
    }

    return 'N/A';
  };

  const getBackend = (session: Session): string => {
    if (!session.config) return 'N/A';
    return session.config.backend;
  };

  const getLastRequestTime = (session: Session): string => {
    const timestamp = session.completed_at || session.created_at;
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
  };

  const filteredSessions = sessions.filter(session =>
    session.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
    session.project.toLowerCase().includes(searchQuery.toLowerCase()) ||
    session.experiment.toLowerCase().includes(searchQuery.toLowerCase()) ||
    getBaseModel(session).toLowerCase().includes(searchQuery.toLowerCase())
  );

  const currentPage = 1;
  const itemsPerPage = 10;
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = Math.min(startIndex + itemsPerPage, filteredSessions.length);
  const paginatedSessions = filteredSessions.slice(startIndex, endIndex);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-4 border-gray-200 border-t-black rounded-full animate-spin" />
          <p className="text-sm text-gray-500">Loading training runs...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 p-8">
        <div className="max-w-2xl mx-auto">
          <div className="bg-gray-50 border border-gray-300 rounded-xl p-6 mb-4">
            <div className="flex items-start gap-3">
              <WarningIcon sx={{ fontSize: 28 }} className="text-black" />
              <div className="flex-1">
                <h3 className="text-sm font-semibold text-black mb-1">Error loading sessions</h3>
                <p className="text-sm text-gray-700">{error}</p>
              </div>
            </div>
          </div>
          <button
            onClick={fetchSessions}
            className="px-4 py-2 bg-black hover:bg-gray-800 text-white text-sm font-medium rounded-lg transition-colors duration-200"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 p-8 overflow-auto">
      <div className="w-full">
        {/* Header */}
        <div className="mb-4">
          <h1 className="text-lg font-semibold text-black mb-2">Training runs</h1>
        </div>

        {/* Search */}
        <div className="mb-6">
          <div className="relative max-w-md">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <SearchIcon sx={{ fontSize: 20 }} className="text-gray-400" />
            </div>
            <input
              type="text"
              placeholder="Search training runs..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-black focus:border-transparent transition-all duration-200 outline-none"
            />
          </div>
        </div>

        {/* Pagination info */}
        <div className="mb-4 flex items-center justify-between">
          <p className="text-sm text-gray-600">
            Showing <span className="font-medium text-black">{startIndex + 1}–{endIndex}</span> of <span className="font-medium text-black">{filteredSessions.length}</span>
          </p>
        </div>

        {/* Table */}
        {paginatedSessions.length === 0 ? (
          <div className="bg-white border border-gray-200 rounded-xl p-12 text-center">
            <div className="flex flex-col items-center gap-4">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center">
                <BarChartIcon sx={{ fontSize: 32 }} className="text-gray-500" />
              </div>
              <div>
                <h3 className="text-sm font-medium text-black mb-1">
                  {searchQuery ? 'No matching training runs' : 'No training runs yet'}
                </h3>
                <p className="text-sm text-gray-500">
                  {searchQuery ? 'Try adjusting your search query.' : 'Start a training session to see it here.'}
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="px-6 py-3.5 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                    Run
                  </th>
                  <th className="px-6 py-3.5 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                    Base Model
                  </th>
                  <th className="px-6 py-3.5 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                    Backend
                  </th>
                  <th className="px-6 py-3.5 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                    Last Request Time
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {paginatedSessions.map((session) => {
                  const projectName = session.project.length > 20 
                    ? session.project.slice(0, 20) + '…' 
                    : session.project;
                  const experimentName = session.experiment.length > 25 
                    ? session.experiment.slice(0, 25) + '…' 
                    : session.experiment;
                  const shortId = session.id.slice(0, 8) + '…';
                  
                  return (
                    <tr
                      key={session.id}
                      onClick={() => navigate(`/runs/${session.id}`)}
                      className="cursor-pointer hover:bg-gray-50 transition-colors duration-150"
                    >
                      <td className="px-6 py-4">
                        <div className="flex flex-col">
                          <span className="text-sm font-medium text-black truncate" title={`${session.project} / ${session.experiment}`}>
                            {projectName} / {experimentName}
                          </span>
                          <code className="text-xs text-gray-500 mt-0.5" title={session.id}>
                            {shortId}
                          </code>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="text-sm text-black">{getBaseModel(session)}</span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="text-sm text-black">{getBackend(session)}</span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="text-sm text-gray-600">{getLastRequestTime(session)}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
