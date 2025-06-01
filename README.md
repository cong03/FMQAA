Code for [From Misleading Queries to Accurate Answers: A Three-Stage Fine-Tuning Method for LLMs](https://arxiv.org/abs/2504.11277), ACL 2025 (Findings)

### Dataset Description

We utilize two specially constructed datasets for training and evaluation:

1. **HaluEval-QA<sub>mis</sub>**: Derived from the QA subset of HaluEval dataset by introducing misleading information into the queries.

2. **CQA<sub>mis<sub>**: Created from CommonsenseQA dataset by incorporating distracting information into potential answers.

### Training Process

Our training consists of three consecutive stages:

1. **Stage 1 - Misinformation Identification**:
   - Trains the model to detect misleading information in queries
   - Labels: "YES" if query contains misinformation, "NO" otherwise

2. **Stage 2 - Misinformation Correction**:
   - Focuses on correcting misleading information
   - Teaches the model to modify and rectify false statements

3. **Stage 3 - Answer Quality Enhancement**:
   - Improves the model's ability to provide high-quality answers
   - Focuses on response accuracy and completeness

All models are trained/fine-tuned using the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework, which provides:
- Efficient fine-tuning capabilities
- Multi-GPU training support
- Quantization options
- Customizable training configurations

## Model Evaluation Script

This script evaluates language models on hallucination detection tasks across three domains: Question Answering (QA), Dialogue, and Summarization. The models being evaluated are trained using the LLaMA-Factory framework with our specialized three-stage training process.

### Args Usage

- `--task`: Specifies the evaluation task type. Options:
  - `qa`: Question Answering evaluation
  - `dialogue`: Dialogue evaluation
  - `summarization`: Summarization evaluation

- `--model`: Specifies the model name/path to evaluate (default: "Qwen2-7B-Instruct"). Note: Models should be trained/fine-tuned using LLaMA-Framework through our three-stage process.

- `--batch_size`: Batch size for evaluation (default: 16)

- `--model_type`: Type of model being evaluated. Options:
  - `qwen`: For Qwen models
  - `llama`: For LLaMA models (trained with LLaMA-Factory)

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
   - Training framework information (LLaMA-Factory)
   - Training stage information (three-stage process)

### Supported Models

The script currently supports:
- Qwen models (through `generate_response_Qwen`)
- LLaMA models (through `generate_response_llama`) - must be trained with LLaMA-Factory using our three-stage process

### Output Metrics

The evaluation calculates:
- Accuracy: correct judgements / total judgements
- Total correct judgements
- Total incorrect judgements
- Framework information (LLaMA-Factory)
- Performance breakdown by training stage

### File Requirements

Before running, you need to prepare:
1. Model checkpoint path (set in code, must be LLaMA-Factory compatible)
2. Instruction prompt file (set in code)
3. Evaluation dataset in JSONL format (set in code)
4. LLaMA-Factory configuration files (if using custom trained models)
5. Stage-specific training data (for three-stage training)

### Notes

- The script automatically handles CUDA device assignment
- Results are saved with timestamp in filename
- Supports both human-annotated and LLM-generated data formats
- All LLaMA models must be trained using LLaMA-Factory framework with our three-stage process
- For Qwen models, standard inference is used but can be adapted to LLaMA-Factory if needed
- Evaluation includes metrics for each training stage's contribution to final performance
