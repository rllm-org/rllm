import { useState, useEffect, useCallback, useRef } from 'react';

export interface Metric {
    id: number;
    step: number;
    data: Record<string, number>;
    created_at: string;
}

interface UseSSEOptions {
    sessionId: string;
    enabled?: boolean;
}

export function useMetricsSSE({ sessionId, enabled = true }: UseSSEOptions) {
    const [metrics, setMetrics] = useState<Metric[]>([]);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<Error | null>(null);
    const seenIds = useRef(new Set<number>());

    useEffect(() => {
        if (!enabled || !sessionId) return;

        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:3000';
        let eventSource: EventSource | null = null;

        // Reset state
        seenIds.current = new Set();
        setMetrics([]);

        // First fetch initial/historical metrics, then connect to SSE
        const initialize = async () => {
            try {
                // Fetch historical metrics
                console.log('[Metrics] Fetching initial metrics...');
                const response = await fetch(`${apiUrl}/api/sessions/${sessionId}/metrics`);
                if (response.ok) {
                    const data: Metric[] = await response.json();
                    console.log('[Metrics] Fetched initial metrics:', data.length);
                    
                    // Track seen IDs
                    data.forEach(m => seenIds.current.add(m.id));
                    setMetrics(data);
                }
            } catch (e) {
                console.error('[Metrics] Failed to fetch initial metrics:', e);
            }

            // Then connect to SSE stream for live updates
            eventSource = new EventSource(
                `${apiUrl}/api/sessions/${sessionId}/metrics/stream`
            );

            eventSource.onopen = () => {
                setIsConnected(true);
                setError(null);
                console.log('[SSE] Connected to metrics stream');
            };

            eventSource.onmessage = (event) => {
                try {
                    const metric: Metric = JSON.parse(event.data);
                    console.log('[SSE] Received metric:', metric);
                    
                    // Avoid duplicates
                    if (!seenIds.current.has(metric.id)) {
                        seenIds.current.add(metric.id);
                        setMetrics((prev) => [...prev, metric]);
                    }
                } catch (e) {
                    console.error('[SSE] Failed to parse metric:', e);
                }
            };

            eventSource.onerror = (e) => {
                console.error('[SSE] Connection error:', e);
                setIsConnected(false);
                setError(new Error('SSE connection failed'));
                eventSource?.close();
            };
        };

        initialize();

        return () => {
            if (eventSource) {
                eventSource.close();
                setIsConnected(false);
            }
        };
    }, [sessionId, enabled]);

    const clearMetrics = useCallback(() => {
        setMetrics([]);
        seenIds.current = new Set();
    }, []);

    return { metrics, isConnected, error, clearMetrics };
}
