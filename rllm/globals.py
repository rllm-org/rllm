"""
Global variables for the RLLM project.
"""

import os
import tempfile

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
REPORT_DIR = os.path.join(tempfile.gettempdir(), "rllm_swebench_reports")

MODEL_NAME_OR_PATH = "rllm"
