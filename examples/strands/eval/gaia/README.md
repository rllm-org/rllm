# Gaia Dataset Evaluation with StrandsAgent

This directory contains the implementation for evaluating the Gaia dataset using the existing Strands + RLLM integration.

## Overview

The Gaia evaluation system leverages the existing `StrandsAgent` and `RLLMModel` integration to evaluate web-based tasks from the Gaia dataset. It provides comprehensive metrics including accuracy, F1 scores, exact match rates, and tool usage statistics.

## Files

- **`gaia_evaluator.py`**: Core evaluation logic and metrics calculation
- **`run_gaia_eval.py`**: Main evaluation script
- **`gaia_config.yaml`**: Configuration file for evaluation parameters
- **`README_gaia_eval.md`**: This documentation file

## Prerequisites

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Download Gaia dataset**:

   ```bash
   python scripts/data/download_gaia.py
   ```

3. **Set up environment variables**:

   ```bash
   # For Together AI
   export TOGETHER_AI_API_KEY="your_api_key"
   export TOGETHER_AI_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct-Turbo"

   # OR for OpenAI
   export OPENAI_API_KEY="your_api_key"
   export OPENAI_BASE_URL="https://api.openai.com/v1"
   export MODEL_NAME="gpt-4o"
   ```

## Usage

### Basic Evaluation

Run evaluation on the entire dataset:

```bash
python run_gaia_eval.py
```

### Custom Configuration

Evaluate with custom parameters:

```bash
python run_gaia_eval.py \
  --dataset_path "path/to/gaia.json" \
  --max_samples 50 \
  --output_dir "custom_output" \
  --output_filename "my_results"
```

### Command Line Arguments

- `--dataset_path`: Path to Gaia dataset JSON file (default: `rllm/data/train/web/gaia.json`)
- `--max_samples`: Maximum number of samples to evaluate (default: all)
- `--output_dir`: Directory to save results (default: `outputs/gaia_eval`)
- `--output_filename`: Base filename for output files (default: auto-generated)

## Output

The evaluation generates two types of output files:

### 1. JSON Results (`{filename}.json`)

Comprehensive evaluation results including:

- Summary statistics (accuracy, F1 scores, execution times)
- Tool usage statistics
- Detailed results for each sample
- Error analysis

### 2. CSV Summary (`{filename}_summary.csv`)

Simplified summary with key metrics:

- Task ID
- Correctness (boolean)
- F1 score
- Exact match (boolean)
- Execution time
- Error messages

## Metrics

### Accuracy Metrics

- **Overall Accuracy**: Percentage of correctly answered questions
- **Exact Match Rate**: Percentage of exact answer matches
- **F1 Score**: Harmonic mean of precision and recall

### Performance Metrics

- **Execution Time**: Time taken to process each sample
- **Tool Usage**: Frequency and type of tool calls
- **Error Rate**: Number of failed evaluations

## Configuration

Edit `gaia_config.yaml` to customize:

- Model parameters (temperature, top_p, max_tokens)
- Tool settings:
  - Search results and timeout for `google_search`
  - Precision for `calculator`
  - Timeout and retry settings for `http_request`
  - File size limits and supported formats for `file_read`
  - Execution timeout and output limits for `python_repl`
- Evaluation thresholds (F1 threshold, exact match bonus)
- Output preferences (save trajectories, logging level)

## Architecture

```
Gaia Dataset → Data Preprocessing → StrandsAgent + Tools → Evaluation Engine → Metrics & Reports
```

- **Data Layer**: Uses existing Gaia dataset loading and preprocessing
- **Agent Layer**: Based on `StrandsAgent` and `RLLMModel`
- **Tool Layer**: Integrates comprehensive tools (search, calculation, HTTP requests, file operations, Python execution)
- **Evaluation Layer**: Implements answer matching and metric calculation
- **Output Layer**: Generates evaluation reports and result files

## Available Tools

The Gaia evaluation system now includes a comprehensive set of tools:

### Core Tools
- **`calculator`**: Mathematical calculations and symbolic math operations
- **`google_search`**: Web search for current information

### Advanced Tools
- **`http_request`**: Make HTTP requests with authentication and session management
- **`file_read`**: Read and analyze files with syntax highlighting
- **`python_repl`**: Execute Python code with state persistence and safety features

### Tool Usage Strategy
- **Primary**: Use `google_search` for information gathering
- **Secondary**: Use specialized tools (`http_request`, `file_read`, `python_repl`) for specific tasks
- **Calculation**: Use `calculator` for mathematical operations

## Integration with Existing Code

This implementation:

- Reuses existing `StrandsAgent` and `RLLMModel` integration
- Leverages comprehensive tool implementations:
  - Custom tools (`google_search`, `calculator`)
  - Native strands tools (`http_request`, `file_read`, `python_repl`)
- Follows the same architecture patterns as `run_strands.py`
- Maintains consistency with existing code style and structure
- Demonstrates the full capabilities of the strands + RLLM integration

## Troubleshooting

### Common Issues

1. **Dataset not found**:

   - Run `python scripts/data/download_gaia.py` first
   - Check the dataset path in configuration

2. **API key errors**:

   - Verify environment variables are set correctly
   - Check API key permissions and quotas

3. **Tool execution failures**:
   - Verify internet connectivity for search tools
   - Check tool-specific error messages

### Debug Mode

Enable verbose logging by setting:

```bash
export LOG_LEVEL=DEBUG
```

## Future Enhancements

- Support for additional evaluation metrics (BLEU, ROUGE)
- Integration with more sophisticated text similarity algorithms
- Batch processing with parallel execution
- Real-time evaluation dashboard
- Integration with VERL engine for training

## Contributing

When modifying the evaluation logic:

1. Update the `GaiaEvaluator` class in `gaia_evaluator.py`
2. Add new metrics to the `EvaluationResult` dataclass
3. Update the configuration file if new parameters are needed
4. Test with a small subset before running full evaluation

## License

This implementation follows the same license as the main RLLM project.
