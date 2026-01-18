import dagre from '@dagrejs/dagre';
import { type Node, type Edge, Position } from '@xyflow/react';

interface Episode {
  id: string;
  task: Record<string, any>;
  is_correct: boolean;
  reward: number | null;
  data: {
    trajectories: Trajectory[];
  };
}

interface Trajectory {
  uid: string;
  name?: string;
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

export interface FlowData {
  nodes: Node[];
  edges: Edge[];
}

export function transformEpisodeToFlow(episode: Episode): FlowData {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  // 1. Create task node
  const taskNodeId = `task-${episode.id}`;
  nodes.push({
    id: taskNodeId,
    type: 'taskNode',
    data: {
      task: episode.task,
      isCorrect: episode.is_correct,
      totalReward: episode.reward
    },
    position: { x: 0, y: 0 }
  });

  // 2. Group trajectories by name to detect parallel vs sequential execution
  const trajectories = episode.data?.trajectories || [];
  const trajectoryGroups: Trajectory[][] = [];

  trajectories.forEach((traj) => {
    const lastGroup = trajectoryGroups[trajectoryGroups.length - 1];
    if (lastGroup && lastGroup[0].name === traj.name) {
      // Same name as previous trajectory → parallel execution
      lastGroup.push(traj);
    } else {
      // Different name or first trajectory → new sequential stage
      trajectoryGroups.push([traj]);
    }
  });

  // 3. Process each group
  let previousStageNodeIds: string[] = [taskNodeId]; // Track last nodes from previous stage

  trajectoryGroups.forEach((group) => {
    const currentStageLastNodeIds: string[] = [];

    group.forEach((traj) => {
      const trajHeaderId = `traj-header-${traj.uid}`;

      // Create trajectory header node
      nodes.push({
        id: trajHeaderId,
        type: 'trajectoryHeaderNode',
        data: {
          name: traj.name || 'Trajectory',
          reward: traj.reward,
          stepCount: traj.steps?.length || 0
        },
        position: { x: 0, y: 0 }
      });

      // Connect to all nodes from previous stage (parallel)
      previousStageNodeIds.forEach((prevNodeId) => {
        edges.push({
          id: `e-${prevNodeId}-${trajHeaderId}`,
          source: prevNodeId,
          target: trajHeaderId,
          type: 'smoothstep',
          animated: false
        });
      });

      // Create step nodes for this trajectory
      let prevStepId = trajHeaderId;
      const steps = traj.steps || [];

      steps.forEach((step, stepIdx) => {
        const stepId = `step-${traj.uid}-${stepIdx}`;

        nodes.push({
          id: stepId,
          type: 'stepNode',
          data: {
            stepIndex: stepIdx,
            observation: step.observation,
            action: step.action,
            reward: step.reward,
            done: step.done,
            chatCompletions: step.chat_completions,
            modelResponse: step.model_response
          },
          position: { x: 0, y: 0 }
        });

        // Edge from previous step
        edges.push({
          id: `e-${prevStepId}-${stepId}`,
          source: prevStepId,
          target: stepId,
          type: 'smoothstep',
          animated: false,
          label: step.reward ? `+${step.reward.toFixed(2)}` : undefined,
          style: {
            stroke: '#9ca3af',
            strokeWidth: 2
          }
        });

        prevStepId = stepId;
      });

      // Track the last node of this trajectory for next stage
      currentStageLastNodeIds.push(prevStepId);
    });

    // Update for next stage
    previousStageNodeIds = currentStageLastNodeIds;
  });

  return { nodes, edges };
}

export function getLayoutedElements(nodes: Node[], edges: Edge[]): FlowData {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));

  // Configure layout
  dagreGraph.setGraph({
    rankdir: 'TB', // Top to Bottom
    nodesep: 80,   // Horizontal spacing between nodes
    ranksep: 100,  // Vertical spacing between ranks
    marginx: 50,
    marginy: 50
  });

  // Set node dimensions
  nodes.forEach((node) => {
    let width = 200;
    let height = 80;

    if (node.type === 'taskNode') {
      width = 400;
      height = 120;
    } else if (node.type === 'trajectoryHeaderNode') {
      width = 200;
      height = 60;
    } else if (node.type === 'stepNode') {
      width = 180;
      height = 70;
    }

    dagreGraph.setNode(node.id, { width, height });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  // Update node positions
  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    const width = node.type === 'taskNode' ? 400 : node.type === 'trajectoryHeaderNode' ? 200 : 180;
    const height = node.type === 'taskNode' ? 120 : node.type === 'trajectoryHeaderNode' ? 60 : 70;

    return {
      ...node,
      position: {
        x: nodeWithPosition.x - width / 2,
        y: nodeWithPosition.y - height / 2
      },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top
    };
  });

  return { nodes: layoutedNodes, edges };
}
