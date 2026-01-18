import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { TrackChangesIcon, SidebarCollapseIcon, SidebarExpandIcon } from './icons';

export const Sidebar: React.FC = () => {
  const location = useLocation();
  const isTrainingRunsActive = location.pathname === '/' || location.pathname.startsWith('/runs');
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <aside className={`
      h-screen bg-white border-r border-gray-200 flex flex-col
      transition-all duration-300 ease-in-out
      ${isCollapsed ? 'w-16' : 'w-56'}
    `}>
      {/* Logo/Header */}
      <div className={`px-4 py-4 flex items-center ${isCollapsed ? 'justify-center' : 'justify-between'}`}>
        {!isCollapsed && (
          <img
            src="/rllm_logo_black.png"
            alt="rLLM"
            className="h-6 w-auto"
          />
        )}
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors"
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {isCollapsed ? (
            <SidebarExpandIcon sx={{ fontSize: 18 }} />
          ) : (
            <SidebarCollapseIcon sx={{ fontSize: 18 }} />
          )}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-2">
        <Link
          to="/"
          className={`
            group flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium
            transition-all duration-200 ease-in-out
            ${isCollapsed ? 'justify-center' : ''}
            ${isTrainingRunsActive
              ? 'bg-gray-100 text-black'
              : 'text-gray-700 hover:bg-gray-50 hover:text-black'
            }
          `}
          title={isCollapsed ? 'Training runs' : undefined}
        >
          <TrackChangesIcon
            sx={{ fontSize: 20 }}
            className={`
              transition-transform duration-200 ease-in-out flex-shrink-0
              ${isTrainingRunsActive ? '' : 'group-hover:scale-110'}
            `}
          />
          {!isCollapsed && (
            <span>Training runs</span>
          )}
        </Link>
      </nav>
    </aside>
  );
};
