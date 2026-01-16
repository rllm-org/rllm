import { Handle, Position } from '@xyflow/react';

export interface TaskNodeData {
  task: any;
  isCorrect: boolean;
  totalReward: number | null;
}

interface TaskNodeProps {
  data: TaskNodeData;
}

export const TaskNode = ({ data }: TaskNodeProps) => {
  const getTaskSummary = (task: any): string => {
    if (typeof task === 'string') return task;
    if (task?.question) return task.question;
    return JSON.stringify(task).slice(0, 100);
  };

  return (
    <div className="px-4 py-3 bg-white border-2 border-black rounded-lg shadow-md min-w-[300px]">
      <Handle type="target" position={Position.Top} className="!bg-gray-400" />

      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-semibold text-gray-600 uppercase">Task</span>
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
          data.isCorrect ? 'bg-black text-white' : 'bg-gray-300 text-black'
        }`}>
          {data.isCorrect ? '✓ Correct' : '✗ Incorrect'}
        </span>
        {data.totalReward !== null && data.totalReward !== undefined && (
          <span className="text-xs text-gray-600">
            Reward: {data.totalReward.toFixed(3)}
          </span>
        )}
      </div>

      <div className="text-sm text-black line-clamp-3">
        {getTaskSummary(data.task)}
      </div>

      <Handle type="source" position={Position.Bottom} className="!bg-gray-400" />
    </div>
  );
};
