import { Handle, Position } from '@xyflow/react';

export interface TrajectoryHeaderNodeData {
  name: string;
  stepCount: number;
  reward: number;
}

interface TrajectoryHeaderNodeProps {
  data: TrajectoryHeaderNodeData;
}

export const TrajectoryHeaderNode = ({ data }: TrajectoryHeaderNodeProps) => {
  return (
    <div className="px-4 py-2 bg-gray-100 border border-gray-300 rounded-lg min-w-[180px]">
      <Handle type="target" position={Position.Top} className="!bg-gray-400" />

      <div className="text-sm font-semibold text-black mb-1">
        {data.name || 'Trajectory'}
      </div>
      <div className="flex items-center justify-between text-xs text-gray-600">
        <span>{data.stepCount} steps</span>
        <span>Reward: {data.reward?.toFixed(3) ?? 'N/A'}</span>
      </div>

      <Handle type="source" position={Position.Bottom} className="!bg-gray-400" />
    </div>
  );
};
