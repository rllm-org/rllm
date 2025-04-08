"""
Global variables for the RLLM project.
"""
# For both vLLM and SGLang samplers
BASE_SAMPLER_PORT = 8080

# Gemini Vertex AI Config (for dataset preprocessing).
GCP_PROJECT_ID = None
GCP_LOCATION = "us-central1"
GEMINI_MODEL = "gemini-1.5-pro-002"
OAI_RM_MODEL = "gpt-4o-mini"

# Reward function constants
THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"