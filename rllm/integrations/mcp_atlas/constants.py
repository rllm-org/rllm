"""Version pins and protocol defaults for MCP-Atlas."""

SOURCE_URL = "https://github.com/scaleapi/mcp-atlas.git"
SOURCE_REVISION = "ab35dcd10cf94985d709265927eec951f5d9faa0"
IMAGE = "ghcr.io/scaleapi/mcp-atlas:1.2.7"
DATASET_REVISION = "b5bcde2"
DATASET_SHA256 = "2d7bc052f14cbcb3b8294293481053f7111d256f9c9deaa96f3ff632d19958d0"
JUDGE_MODEL = "openrouter/google/gemini-2.5-pro"
JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
DIRECT_GEMINI_JUDGE_MODEL = "gemini/gemini-2.5-pro"
MAX_TURNS = 256
MAX_TOOL_CALLS = 100
TASK_TIMEOUT_SECONDS = 1800
DEFAULT_CONCURRENCY = 5
GATEWAY_API_KEY = "sk-rllm-gateway"

# Servers in the pinned image that work without user-provided credentials.
# Keep this explicit instead of using the upstream test_servers.py detector:
# that detector's ``[A-Z_]+`` environment-variable regex misses the digit in
# E2B_API_KEY and therefore incorrectly reports e2b-server as keyless.
DEFAULT_SERVERS = frozenset(
    {
        "arxiv",
        "calculator",
        "cli-mcp-server",
        "clinicaltrialsgov-mcp-server",
        "context7",
        "ddg-search",
        "desktop-commander",
        "fetch",
        "filesystem",
        "git",
        "mcp-code-executor",
        "mcp-server-code-runner",
        "memory",
        "met-museum",
        "open-library",
        "osm-mcp-server",
        "pubmed",
        "weather",
        "whois",
        "wikipedia",
    }
)
