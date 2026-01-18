import { useState } from 'react';

interface Episode {
    id: string;
    session_id: string;
    step: number;
    task: Record<string, any>;
    is_correct: boolean;
    reward: number | null;
    data: {
        trajectories: Trajectory[];
    };
    created_at: string;
}

interface Trajectory {
    uid: string;
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

interface EpisodeViewerProps {
    sessionId?: string;
}

/**
 * EpisodeViewer component - displays episode data with trajectories.
 * MVP Step 2: Raw JSON display of episodes for debugging.
 */
export function EpisodeViewer({ sessionId }: EpisodeViewerProps) {
    const [localSessionId, setLocalSessionId] = useState(sessionId || '');
    const [episodes, setEpisodes] = useState<Episode[]>([]);
    const [selectedEpisode, setSelectedEpisode] = useState<Episode | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchEpisodes = async () => {
        if (!localSessionId) return;

        setLoading(true);
        setError(null);

        try {
            const response = await fetch(
                `http://localhost:3000/api/episodes?session_id=${localSessionId}`
            );

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            setEpisodes(data);
        } catch (err: any) {
            setError(err.message);
            setEpisodes([]);
        } finally {
            setLoading(false);
        }
    };

    const fetchEpisodeDetails = async (episodeId: string) => {
        try {
            const response = await fetch(
                `http://localhost:3000/api/episodes/${episodeId}`
            );

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            setSelectedEpisode(data);
        } catch (err: any) {
            setError(err.message);
        }
    };

    return (
        <div style={{ padding: '20px', fontFamily: 'monospace' }}>
            <h1>Episode Viewer (MVP Step 2)</h1>

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
                <button onClick={fetchEpisodes} disabled={loading || !localSessionId}>
                    {loading ? 'Loading...' : 'Load Episodes'}
                </button>
            </div>

            {error && (
                <div style={{ padding: '10px', background: '#ffebee', color: '#c62828', marginBottom: '20px' }}>
                    Error: {error}
                </div>
            )}

            <div style={{ display: 'flex', gap: '20px' }}>
                {/* Episode List */}
                <div style={{ flex: 1 }}>
                    <h2>Episodes ({episodes.length})</h2>
                    <div
                        style={{
                            background: '#f5f5f5',
                            padding: '10px',
                            borderRadius: '5px',
                            maxHeight: '600px',
                            overflowY: 'auto',
                        }}
                    >
                        {episodes.length === 0 ? (
                            <div style={{ color: '#888' }}>
                                No episodes loaded. Enter a session ID and click "Load Episodes".
                            </div>
                        ) : (
                            episodes.map((episode) => (
                                <div
                                    key={episode.id}
                                    onClick={() => fetchEpisodeDetails(episode.id)}
                                    style={{
                                        padding: '10px',
                                        marginBottom: '10px',
                                        background: selectedEpisode?.id === episode.id ? '#e3f2fd' : 'white',
                                        border: '1px solid #ddd',
                                        borderRadius: '3px',
                                        cursor: 'pointer',
                                    }}
                                >
                                    <div style={{ fontWeight: 'bold' }}>
                                        {episode.id}
                                        <span
                                            style={{
                                                marginLeft: '10px',
                                                padding: '2px 6px',
                                                background: episode.is_correct ? '#c8e6c9' : '#ffcdd2',
                                                borderRadius: '3px',
                                                fontSize: '12px',
                                            }}
                                        >
                                            {episode.is_correct ? '✓ Correct' : '✗ Incorrect'}
                                        </span>
                                    </div>
                                    <div style={{ fontSize: '12px', color: '#666' }}>
                                        Step: {episode.step} | Reward: {episode.reward ?? 'N/A'}
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* Episode Details */}
                <div style={{ flex: 2 }}>
                    <h2>Episode Details</h2>
                    <div
                        style={{
                            background: '#1e1e1e',
                            color: '#d4d4d4',
                            padding: '15px',
                            borderRadius: '5px',
                            maxHeight: '600px',
                            overflowY: 'auto',
                            fontSize: '14px',
                        }}
                    >
                        {selectedEpisode ? (
                            <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                                {JSON.stringify(selectedEpisode, null, 2)}
                            </pre>
                        ) : (
                            <div style={{ color: '#888' }}>
                                Click an episode to view details
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <div style={{ marginTop: '20px', color: '#666', fontSize: '12px' }}>
                <strong>Note:</strong> This is MVP Step 2 - displaying raw episode data for debugging.
                Step 3 will add proper UI components and visualization.
            </div>
        </div>
    );
}

export default EpisodeViewer;
