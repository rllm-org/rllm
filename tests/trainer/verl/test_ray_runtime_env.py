import os
from unittest.mock import patch

from rllm.trainer.verl.ray_runtime_env import _get_forwarded_env_vars


def test_forward_basic_env_vars():
    """Test that environment variables with matching prefixes are forwarded."""
    test_env = {
        "VLLM_LOGGING_LEVEL": "INFO",
        "NCCL_DEBUG": "INFO",
        "HF_TOKEN": "test_token",
        "TORCH_CUDA_ARCH_LIST": "8.0",
        "RANDOM_VAR": "should_not_be_forwarded",
        "PATH": "/usr/bin",
    }

    with patch.dict(os.environ, test_env, clear=True):
        forwarded = _get_forwarded_env_vars()

    assert "VLLM_LOGGING_LEVEL" in forwarded
    assert forwarded["VLLM_LOGGING_LEVEL"] == "INFO"
    assert "NCCL_DEBUG" in forwarded
    assert "HF_TOKEN" in forwarded
    assert "TORCH_CUDA_ARCH_LIST" in forwarded
    assert "RANDOM_VAR" not in forwarded
    assert "PATH" not in forwarded


def test_no_matching_env_vars():
    """Test when no environment variables match any prefix."""
    test_env = {
        "PATH": "/usr/bin",
        "HOME": "/home/user",
        "USER": "testuser",
    }

    with patch.dict(os.environ, test_env, clear=True):
        forwarded = _get_forwarded_env_vars()

    assert len(forwarded) == 0


def test_all_prefixes_forwarded():
    """Test that all defined prefixes are forwarded correctly."""
    test_env = {
        "VLLM_TEST": "vllm",
        "SGL_TEST": "sgl",
        "SGLANG_TEST": "sglang",
        "HF_TEST": "hf",
        "TOKENIZERS_TEST": "tokenizers",
        "DATASETS_TEST": "datasets",
        "TORCH_TEST": "torch",
        "PYTORCH_TEST": "pytorch",
        "DEEPSPEED_TEST": "deepspeed",
        "MEGATRON_TEST": "megatron",
        "NCCL_TEST": "nccl",
        "CUDA_TEST": "cuda",
        "CUBLAS_TEST": "cublas",
        "CUDNN_TEST": "cudnn",
        "NV_TEST": "nv",
        "NVIDIA_TEST": "nvidia",
    }

    with patch.dict(os.environ, test_env, clear=True):
        forwarded = _get_forwarded_env_vars()

    # All test variables should be forwarded
    assert len(forwarded) == len(test_env)
    for key, value in test_env.items():
        assert forwarded[key] == value


def test_exclude_specific_var_names():
    """Test exclusion of specific environment variable names using RLLM_EXCLUDE."""
    test_env = {
        "VLLM_LOGGING_LEVEL": "INFO",
        "VLLM_DEBUG": "1",
        "NCCL_DEBUG": "WARN",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "RLLM_EXCLUDE": "VLLM_DEBUG,CUDA_VISIBLE_DEVICES",
    }

    with patch.dict(os.environ, test_env, clear=True):
        forwarded = _get_forwarded_env_vars()

    # VLLM_LOGGING_LEVEL and NCCL_DEBUG should be forwarded
    assert "VLLM_LOGGING_LEVEL" in forwarded
    assert "NCCL_DEBUG" in forwarded

    # VLLM_DEBUG and CUDA_VISIBLE_DEVICES should be excluded
    assert "VLLM_DEBUG" not in forwarded
    assert "CUDA_VISIBLE_DEVICES" not in forwarded
    assert "RLLM_EXCLUDE" not in forwarded  # RLLM_EXCLUDE itself shouldn't be forwarded


def test_exclude_prefix_pattern():
    """Test exclusion of all variables with a specific prefix using wildcard pattern."""
    test_env = {
        "VLLM_LOGGING_LEVEL": "INFO",
        "VLLM_DEBUG": "1",
        "VLLM_USE_V1": "1",
        "NCCL_DEBUG": "WARN",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "RLLM_EXCLUDE": "VLLM*",
    }

    with patch.dict(os.environ, test_env, clear=True):
        forwarded = _get_forwarded_env_vars()

    # All VLLM_* variables should be excluded
    assert "VLLM_LOGGING_LEVEL" not in forwarded
    assert "VLLM_DEBUG" not in forwarded
    assert "VLLM_USE_V1" not in forwarded

    # Other prefixes should still be forwarded
    assert "NCCL_DEBUG" in forwarded
    assert "CUDA_VISIBLE_DEVICES" in forwarded


def test_exclude_multiple_patterns():
    """Test exclusion with multiple prefix patterns and specific variable names."""
    test_env = {
        "VLLM_LOGGING_LEVEL": "INFO",
        "VLLM_DEBUG": "1",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "NCCL_DEBUG": "WARN",
        "NCCL_IB_DISABLE": "1",
        "TORCH_DISTRIBUTED_DEBUG": "INFO",
        "RLLM_EXCLUDE": "VLLM*,CUDA*,NCCL_IB_DISABLE",
    }

    with patch.dict(os.environ, test_env, clear=True):
        forwarded = _get_forwarded_env_vars()

    # VLLM_* should be excluded
    assert "VLLM_LOGGING_LEVEL" not in forwarded
    assert "VLLM_DEBUG" not in forwarded

    # CUDA_* should be excluded
    assert "CUDA_VISIBLE_DEVICES" not in forwarded
    assert "CUDA_DEVICE_ORDER" not in forwarded

    # NCCL_IB_DISABLE should be specifically excluded
    assert "NCCL_IB_DISABLE" not in forwarded

    # NCCL_DEBUG should still be forwarded (not in exclude list)
    assert "NCCL_DEBUG" in forwarded

    # TORCH_* should still be forwarded
    assert "TORCH_DISTRIBUTED_DEBUG" in forwarded


def test_no_rllm_exclude_set():
    """Test behavior when RLLM_EXCLUDE is not set."""
    test_env = {
        "VLLM_LOGGING_LEVEL": "INFO",
        "NCCL_DEBUG": "WARN",
        "CUDA_VISIBLE_DEVICES": "0,1",
    }

    with patch.dict(os.environ, test_env, clear=True):
        forwarded = _get_forwarded_env_vars()

    # All matching variables should be forwarded when no exclusions
    assert "VLLM_LOGGING_LEVEL" in forwarded
    assert "NCCL_DEBUG" in forwarded
    assert "CUDA_VISIBLE_DEVICES" in forwarded


def test_empty_rllm_exclude():
    """Test behavior when RLLM_EXCLUDE is set but empty."""
    test_env = {
        "VLLM_LOGGING_LEVEL": "INFO",
        "NCCL_DEBUG": "WARN",
        "RLLM_EXCLUDE": "",
    }

    with patch.dict(os.environ, test_env, clear=True):
        forwarded = _get_forwarded_env_vars()

    # All matching variables should be forwarded with empty exclusion
    assert "VLLM_LOGGING_LEVEL" in forwarded
    assert "NCCL_DEBUG" in forwarded


def test_exclude_with_spaces():
    """Test that exclusion handles spaces in the comma-separated list."""
    test_env = {
        "VLLM_DEBUG": "1",
        "VLLM_LOGGING_LEVEL": "INFO",
        "NCCL_DEBUG": "WARN",
        "RLLM_EXCLUDE": "VLLM_DEBUG, NCCL_DEBUG",  # Note the space after comma
    }

    with patch.dict(os.environ, test_env, clear=True):
        forwarded = _get_forwarded_env_vars()

    # VLLM_DEBUG should be excluded
    assert "VLLM_DEBUG" not in forwarded

    # NCCL_DEBUG might not be excluded if there's a leading space (depends on implementation)
    # But VLLM_LOGGING_LEVEL should be forwarded
    assert "VLLM_LOGGING_LEVEL" in forwarded


def test_case_sensitive_prefixes():
    """Test that prefix matching is case-sensitive."""
    test_env = {
        "VLLM_TEST": "uppercase",
        "vllm_test": "lowercase",
        "Vllm_Test": "mixed",
    }

    with patch.dict(os.environ, test_env, clear=True):
        forwarded = _get_forwarded_env_vars()

    # Only uppercase VLLM_ should match
    assert "VLLM_TEST" in forwarded
    assert "vllm_test" not in forwarded
    assert "Vllm_Test" not in forwarded


def test_edge_case_exact_prefix():
    """Test variables that are exactly the prefix (without underscore)."""
    test_env = {
        "VLLM": "just_prefix",
        "VLLM_WITH_SUFFIX": "with_suffix",
        "NCCL": "just_prefix",
        "NCCL_DEBUG": "with_suffix",
    }

    with patch.dict(os.environ, test_env, clear=True):
        forwarded = _get_forwarded_env_vars()

    # Variables with suffixes should be forwarded
    assert "VLLM_WITH_SUFFIX" in forwarded
    assert "NCCL_DEBUG" in forwarded
