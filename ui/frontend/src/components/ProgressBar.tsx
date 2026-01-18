import React from 'react';

interface ProgressBarProps {
  doneFrac: number;
  currentEpoch: number;
  totalEpochs: number;
  currentBatch: number;
  totalBatches: number | null;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  doneFrac,
  currentEpoch,
  totalEpochs,
  currentBatch,
  totalBatches,
}) => {
  const percentage = Math.min(Math.round(doneFrac * 100), 100);

  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2 text-xs text-gray-600">
        <span>Epoch {currentEpoch + 1}/{totalEpochs}</span>
        {totalBatches && (
          <>
            <span className="text-gray-300">|</span>
            <span>Batch {currentBatch + 1}/{totalBatches}</span>
          </>
        )}
      </div>

      <div className="flex items-center gap-2">
        <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all duration-300"
            style={{ width: `${percentage}%` }}
          />
        </div>
        <span className="text-xs font-semibold text-black w-8">{percentage}%</span>
      </div>
    </div>
  );
};
