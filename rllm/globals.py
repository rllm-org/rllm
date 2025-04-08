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

# SWEBench Harness Config
SWEBENCH_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
MAX_WORKERS = 4
FORCE_REBUILD = False
CACHE_LEVEL = "None"
CLEAN = False
OPEN_FILE_LIMIT = 4096
TIMEOUT = 1_800
NAMESPACE = None
REWRITE_REPORTS = False
SPLIT = "test"
INSTANCE_IMAGE_TAG = "latest"
REPORT_DIR = "."

MODEL_NAME_OR_PATH = "rllm"
