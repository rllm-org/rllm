# AppWorld Agent Examples

Examples for training and running AppWorld agents using rLLM. AppWorld provides a multi-application environment with 9 applications (Spotify, Gmail, Calendar, etc.) and 457 APIs.

## Quick Start

### 1. Environment Setup

```bash
# CPU-only (for local development/testing)
bash scripts/agent/appworld/install_env_cpu.sh

# GPU (for training/production)
bash scripts/agent/appworld/install_env_gpu.sh
```

**Note**: For GPU setup, adjust CUDA version in the script if needed.

### 2. Install AppWorld

After setting up the environment:

```bash
conda activate rllm_py311  # or rllm_py311_gpu
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
appworld download data --root /path/to/your/data/directory
appworld verify tasks
```


## References

- [AppWorld GitHub](https://github.com/StonyBrookNLP/appworld)
- [rLLM Documentation](../../docs/)
