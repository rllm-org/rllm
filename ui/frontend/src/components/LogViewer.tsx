import { useState, useEffect } from 'react';
import { useMetricsSSE } from '../hooks/useSSE';

interface LogViewerProps {
    sessionId?: string;
}

/**
 * LogViewer component - displays raw log data from SSE stream.
 * MVP Step 1: Simple text display of incoming metrics.
 */
export function LogViewer({ sessionId }: LogViewerProps) {
    const [localSessionId, setLocalSessionId] = useState(sessionId || '');
    const [isWatching, setIsWatching] = useState(false);

    const { metrics, isConnected, error, clearMetrics } = useMetricsSSE({
        sessionId: localSessionId,
        enabled: isWatching,
    });

    // Auto-scroll to bottom when new metrics arrive
    useEffect(() => {
        const logContainer = document.getElementById('log-container');
        if (logContainer) {
            logContainer.scrollTop = logContainer.scrollHeight;
        }
    }, [metrics]);

    return (
        <div style={{ padding: '20px', fontFamily: 'monospace' }}>
            <h1>rLLM UI - Log Viewer (MVP Step 1)</h1>

            <div style={{ marginBottom: '20px' }}>
                <label>
                    Session ID:{' '}
                    <input
                        type="text"
                        value={localSessionId}
                        onChange={(e) => setLocalSessionId(e.target.value)}
                        placeholder="Enter session ID"
                        style={{ width: '300px', padding: '5px' }}
                    />
                </label>
                {' '}
                <button onClick={() => setIsWatching(!isWatching)}>
                    {isWatching ? 'Stop Watching' : 'Start Watching'}
                </button>
                {' '}
                <button onClick={clearMetrics}>Clear Logs</button>
            </div>

            <div style={{ marginBottom: '10px' }}>
                <strong>Status:</strong>{' '}
                <span style={{ color: isConnected ? 'green' : 'red' }}>
                    {isConnected ? '● Connected' : '○ Disconnected'}
                </span>
                {error && <span style={{ color: 'red' }}> - {error.message}</span>}
            </div>

            <div
                id="log-container"
                style={{
                    background: '#1e1e1e',
                    color: '#d4d4d4',
                    padding: '15px',
                    borderRadius: '5px',
                    height: '400px',
                    overflowY: 'auto',
                    fontSize: '14px',
                }}
            >
                {metrics.length === 0 ? (
                    <div style={{ color: '#888' }}>
                        {isWatching
                            ? 'Waiting for metrics...'
                            : 'Enter a session ID and click "Start Watching" to begin.'}
                    </div>
                ) : (
                    metrics.map((metric, index) => (
                        <div key={metric.id || index} style={{ marginBottom: '8px' }}>
                            <span style={{ color: '#569cd6' }}>[Step {metric.step}]</span>{' '}
                            <span style={{ color: '#dcdcaa' }}>
                                {JSON.stringify(metric.data)}
                            </span>
                        </div>
                    ))
                )}
            </div>

            <div style={{ marginTop: '20px', color: '#666', fontSize: '12px' }}>
                <strong>Raw Metrics Count:</strong> {metrics.length}
            </div>
        </div>
    );
}

export default LogViewer;
