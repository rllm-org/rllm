import React from 'react';
import { ChevronLeftIcon } from '../icons';

interface StepDetailPanelProps {
  step: any;
  onClose: () => void;
}

export const StepDetailPanel: React.FC<StepDetailPanelProps> = ({ step, onClose }) => {
  const renderValue = (value: any): React.ReactNode => {
    if (value === null || value === undefined) return <span className="text-gray-500">N/A</span>;
    if (typeof value === 'object') {
      return <pre className="text-xs bg-gray-50 p-2 rounded overflow-x-auto">{JSON.stringify(value, null, 2)}</pre>;
    }
    return <span className="text-sm text-black">{String(value)}</span>;
  };

  return (
    <div className="absolute right-0 top-0 h-full w-96 bg-white border-l border-gray-300 shadow-xl overflow-y-auto z-10">
      <div className="p-4">
        <button
          onClick={onClose}
          className="mb-4 flex items-center gap-2 text-gray-600 hover:text-black transition-colors"
        >
          <ChevronLeftIcon sx={{ fontSize: 20 }} />
          <span className="text-sm">Back</span>
        </button>

        <h3 className="text-lg font-semibold text-black mb-4">
          Step {step.stepIndex + 1} Details
        </h3>

        <div className="space-y-4">
          <div>
            <h4 className="text-sm font-medium text-gray-600 mb-1">Reward</h4>
            {renderValue(step.reward)}
          </div>

          {step.observation !== null && step.observation !== undefined && (
            <div>
              <h4 className="text-sm font-medium text-gray-600 mb-1">Observation</h4>
              {renderValue(step.observation)}
            </div>
          )}

          {step.action && (
            <div>
              <h4 className="text-sm font-medium text-gray-600 mb-1">Action</h4>
              {renderValue(step.action)}
            </div>
          )}

          {step.modelResponse && (
            <div>
              <h4 className="text-sm font-medium text-gray-600 mb-1">Model Response</h4>
              {renderValue(step.modelResponse)}
            </div>
          )}

          {step.chatCompletions && (
            <div>
              <h4 className="text-sm font-medium text-gray-600 mb-1">Chat Completions</h4>
              {renderValue(step.chatCompletions)}
            </div>
          )}

          <div>
            <h4 className="text-sm font-medium text-gray-600 mb-1">Done</h4>
            <span className={`inline-block px-2 py-0.5 rounded text-xs ${
              step.done ? 'bg-black text-white' : 'bg-gray-200 text-black'
            }`}>
              {step.done ? 'True' : 'False'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
