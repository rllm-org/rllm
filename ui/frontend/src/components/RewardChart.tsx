import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { Metric } from '../hooks/useSSE';

interface ChartDataPoint {
  step: number;
  value?: number;
}

interface MetricChartProps {
  metrics: Metric[];
  selectedMetric: string;
  selectedStep: number | null;
  onStepClick: (step: number) => void;
}

/**
 * MetricChart - Displays any selected metric over training steps.
 * Clicking on a data point selects that step to show episodes.
 */
export const RewardChart: React.FC<MetricChartProps> = ({
  metrics,
  selectedMetric,
  selectedStep,
  onStepClick,
}) => {
  // Transform metrics into chart data points for the selected metric
  const chartData: ChartDataPoint[] = React.useMemo(() => {
    const dataMap = new Map<number, ChartDataPoint>();

    metrics.forEach((metric) => {
      const step = metric.step;
      const existing = dataMap.get(step) || { step };

      // Extract the selected metric from the data object
      if (metric.data[selectedMetric] !== undefined) {
        existing.value = metric.data[selectedMetric];
      }

      dataMap.set(step, existing);
    });

    // Sort by step and return as array
    return Array.from(dataMap.values()).sort((a, b) => a.step - b.step);
  }, [metrics, selectedMetric]);

  // Custom click handler for the chart area
  const handleChartClick = (data: any) => {
    // Try activePayload first
    if (data && data.activePayload && data.activePayload.length > 0) {
      const step = data.activePayload[0].payload.step;
      onStepClick(step);
      return;
    }
    
    // Fallback to activeLabel (which is the step value on x-axis)
    if (data && data.activeLabel !== undefined) {
      onStepClick(data.activeLabel);
    }
  };

  // Click handler for individual dots
  const handleDotClick = (data: any) => {
    if (data && data.payload) {
      onStepClick(data.payload.step);
    }
  };

  // Check if we have any data to show
  const hasData = chartData.some((d) => d.value !== undefined);

  if (chartData.length === 0 || !hasData) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <div className="text-center">
          <p className="text-sm">No data for "{selectedMetric}"</p>
          <p className="text-xs text-gray-400 mt-1">
            Try selecting a different metric
          </p>
        </div>
      </div>
    );
  }

  // Get Y-axis label from metric name (e.g., "reward/mean" -> "Reward")
  const yAxisLabel = selectedMetric.split('/')[0].charAt(0).toUpperCase() + 
                     selectedMetric.split('/')[0].slice(1);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart
        data={chartData}
        onClick={handleChartClick}
        margin={{ top: 10, right: 30, left: 10, bottom: 10 }}
        style={{ cursor: 'crosshair', outline: 'none' }}
        accessibilityLayer={false}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="step"
          tick={{ fontSize: 12, fill: '#6b7280' }}
          tickLine={{ stroke: '#d1d5db' }}
          axisLine={{ stroke: '#d1d5db' }}
          label={{
            value: 'Step',
            position: 'insideBottomRight',
            offset: -5,
            fontSize: 12,
            fill: '#6b7280',
          }}
        />
        <YAxis
          tick={{ fontSize: 12, fill: '#6b7280' }}
          tickLine={{ stroke: '#d1d5db' }}
          axisLine={{ stroke: '#d1d5db' }}
          label={{
            value: yAxisLabel,
            angle: -90,
            position: 'insideLeft',
            fontSize: 12,
            fill: '#6b7280',
          }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'white',
            border: '1px solid #e5e7eb',
            borderRadius: '8px',
            boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
            fontSize: '12px',
          }}
          labelStyle={{ fontWeight: 600, marginBottom: '4px' }}
          formatter={(value) => [
            typeof value === 'number' ? value.toFixed(4) : 'N/A',
            selectedMetric,
          ]}
          labelFormatter={(step) => `Step ${step}`}
        />
        <Legend
          formatter={() => selectedMetric}
          wrapperStyle={{ fontSize: '12px' }}
        />

        {/* Reference line for selected step */}
        {selectedStep !== null && (
          <ReferenceLine
            x={selectedStep}
            stroke="#000"
            strokeWidth={2}
            strokeDasharray="4 4"
          />
        )}

        {/* Metric line */}
        <Line
          type="monotone"
          dataKey="value"
          stroke="#2563eb"
          strokeWidth={2}
          dot={(props: any) => {
            const { cx, cy, payload } = props;
            if (cx === undefined || cy === undefined) return null;
            return (
              <circle
                cx={cx}
                cy={cy}
                r={4}
                fill="#2563eb"
                cursor="pointer"
                onClick={() => onStepClick(payload.step)}
                style={{ pointerEvents: 'all' }}
              />
            );
          }}
          activeDot={{
            r: 8,
            fill: '#1d4ed8',
            stroke: '#fff',
            strokeWidth: 2,
            cursor: 'pointer',
            onClick: handleDotClick,
          }}
          connectNulls
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

/**
 * Extract all unique metric keys from the metrics data
 */
export function getAvailableMetrics(metrics: Metric[]): string[] {
  const metricSet = new Set<string>();
  
  metrics.forEach((metric) => {
    Object.keys(metric.data).forEach((key) => {
      metricSet.add(key);
    });
  });
  
  // Sort alphabetically, but put reward/mean first if it exists
  const sorted = Array.from(metricSet).sort((a, b) => {
    if (a === 'reward/mean') return -1;
    if (b === 'reward/mean') return 1;
    return a.localeCompare(b);
  });
  
  return sorted;
}
