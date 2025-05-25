# Command Usages

## Model Evaluation Script

This script evaluates language models on hallucination detection tasks across three domains: Question Answering (QA), Dialogue, and Summarization.

### Args Usage

- `--task`: Specifies the evaluation task type. Options:
  - `qa`: Question Answering evaluation
  - `dialogue`: Dialogue evaluation
  - `summarization`: Summarization evaluation

- `--model`: Specifies the model name/path to evaluate (default: "Qwen2-7B-Instruct")

- `--batch_size`: Batch size for evaluation (default: 16)

- `--model_type`: Type of model being evaluated. Options:
  - `qwen`: For Qwen models
  - `llama`: For LLaMA models

### Evaluation Process

1. For each sample in the dataset:
   - Randomly selects either the hallucinated or correct response (50/50 chance)
   - Presents the sample to the model with an instruction prompt
   - Records whether the model correctly identifies if the response contains hallucination

2. Outputs results in JSONL format including:
   - Input sample
   - Ground truth label
   - Model's judgement
   - Final accuracy metrics

### Supported Models

The script currently supports:
- Qwen models (through `generate_response_Qwen`)
- LLaMA models (through `generate_response_llama`)

### Output Metrics

The evaluation calculates:
- Accuracy: correct judgements / total judgements
- Total correct judgements
- Total incorrect judgements

### File Requirements

Before running, you need to prepare:
1. Model checkpoint path (set in code)
2. Instruction prompt file (set in code)
3. Evaluation dataset in JSONL format (set in code)

### Notes

- The script automatically handles CUDA device assignment
- Results are saved with timestamp in filename
- Supports both human-annotated and LLM-generated data formats