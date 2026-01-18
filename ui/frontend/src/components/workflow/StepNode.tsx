import { Handle, Position } from '@xyflow/react';

export interface StepNodeData {
  stepIndex: number;
  observation: any;
  action: any;
  reward: number;
  done: boolean;
  chatCompletions?: any;
  modelResponse?: any;
}

interface StepNodeProps {
  data: StepNodeData;
  selected?: boolean;
}

export const StepNode = ({ data, selected }: StepNodeProps) => {
  return (
    <div
      className={`px-3 py-2 bg-white border rounded-lg min-w-[160px] transition-all cursor-pointer ${
        selected ? 'border-black border-2 shadow-lg' : 'border-gray-300 hover:border-gray-400'
      }`}
    >
      <Handle type="target" position={Position.Top} className="!bg-gray-400" />

      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-medium text-gray-600">
          Step {data.stepIndex + 1}
        </span>
        {data.done && (
          <span className="text-xs px-1.5 py-0.5 bg-gray-200 text-black rounded">
            Done
          </span>
        )}
      </div>

      <div className="text-sm text-black font-medium">
        Reward: {data.reward?.toFixed(3) ?? 'N/A'}
      </div>

      {data.action && (
        <div className="text-xs text-gray-600 mt-1 line-clamp-2">
          {typeof data.action === 'string' ? data.action : JSON.stringify(data.action).slice(0, 50)}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} className="!bg-gray-400" />
    </div>
  );
};
