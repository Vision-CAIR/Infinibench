# InfiniBench Evaluation

This module provides automated evaluation capabilities for the InfiniBench video-based question-answering benchmark. It supports both open-ended and multiple-choice question evaluation using GPT-4o-mini scoring and accuracy metrics respectively.
# Running Challenge

## Test Set Evaluation
Submit your predictions to the [2025 ICCV CLVL - Long Video Understanding Challenge (InfiniBench)](https://www.codabench.org/competitions/10065/). The evaluation will be performed automatically on the Codabench platform. Please follow the guidelines provided in the challenge description.

## Validation Set Evaluation
Follow the instructions below to run the evaluation script locally.

> **Note:** Test set ground truth is not publicly available. Evaluation is based on the predictions you provide during challenge submission. Ensure your predictions follow the correct format as specified in the challenge guidelines.
## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## Requirements

- Python 3.7+
- OpenAI API key (for open-ended evaluation)
- Required Python packages:
  ```
  openai>=1.0.0
  ```

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/Vision-CAIR/Infinibench.git
   cd Infinibench/evaluation
   ```

2. **Install dependencies**:
   ```bash
   pip install openai
   ```

3. **Set up OpenAI API key**:
   ```bash
   # Linux/Mac
   export OPENAI_API_KEY="your-openai-api-key-here"
   
   # Windows (PowerShell)
   $env:OPENAI_API_KEY="your-openai-api-key-here"
   
   # Windows (Command Prompt)
   set OPENAI_API_KEY=your-openai-api-key-here
   ```

## Data Preparation

### Directory Structure

Organize your prediction files in a single directory:

```
predictions/
├── summarization.json              # Open-ended skill
├── spoiler_questions.json          # Open-ended skill
├── deep_context_understanding.json # Open-ended skill
├── linking_multiple_events.json    # Open-ended skill
├── character_actions.json          # MCQ skill
├── scene_transitions.json          # MCQ skill
├── choronological_understanding.json # MCQ skill
└── global_appearance.json          # MCQ skill
```

### Input Data Format

Your prediction files must be in JSON format with the following structure:

#### For Open-Ended Questions:
```json
[
  {
    "question": "What is happening in this scene?",
    "answer": "The character is walking through a forest during sunset.",
    "pred": "A person is walking in a wooded area at dusk."
  },
  {
    "question": "Describe the character's emotions.",
    "answer": "The character appears sad and contemplative.",
    "pred": "The person looks thoughtful and melancholy."
  }
]
```

#### For Multiple-Choice Questions:
```json
[
  {
    "question": "What color is the character's shirt?",
    "answer_idx": 2,
    "pred": 2,
  },
  {
    "question": "Where does this scene take place?",
    "answer_idx": 0,
    "pred": 1,
  }
]
```


### Required Fields

**Open-ended questions** must have:
- `question`: The question text
- `answer`: The ground truth answer
- `pred`: The predicted answer

**Multiple-choice questions** must have:
- `question`: The question text
- `answer_idx`: The correct answer index (0-based)
- `pred`: The predicted answer index (0-based)

## Usage

### Basic Usage

```bash
python eval_script.py --pred_dir /path/to/predictions
```

### Advanced Usage

```bash
# Custom thread count and batch size
python eval_script.py --pred_dir ./predictions --max_threads 8 --batch_size 10

# Skip specific evaluation types
python eval_script.py --pred_dir ./predictions --skip_mcq
python eval_script.py --pred_dir ./predictions --skip_open_ended

# Enable debug logging
python eval_script.py --pred_dir ./predictions --log_level DEBUG
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pred_dir` | str | **Required** | Directory containing prediction JSON files |
| `--max_threads` | int | 4 | Maximum number of threads for parallel processing |
| `--batch_size` | int | 10 | Number of items to process before saving (safety feature) |
| `--skip_open_ended` | flag | False | Skip evaluation of open-ended skills |
| `--skip_mcq` | flag | False | Skip evaluation of multiple-choice questions |
| `--log_level` | str | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |



### Log Files

Detailed logs are saved to `evaluation.log`:
- Timestamps for all operations
- Error messages and warnings
- Batch processing progress
- Final statistics

## Troubleshooting

### Common Issues

#### 1. OpenAI API Key Not Found
```
Error: OPENAI_API_KEY environment variable is not set
```
**Solution**: Set your OpenAI API key as an environment variable.

#### 2. Missing Required Fields
```
Warning: Missing required fields: ['pred']
```
**Solution**: Ensure all JSON files have the required fields as specified in [Data Preparation](#data-preparation).

#### 3. File Not Found
```
Error: File not found: /path/to/predictions/skill.json
```
**Solution**: Check that the prediction directory path is correct and files exist.

#### 4. Invalid JSON Format
```
Error: Invalid JSON in file skill.json: Expecting ',' delimiter
```
**Solution**: Validate your JSON files using a JSON validator.

#### 5. Rate Limiting
```
Error: Rate limit exceeded
```
**Solution**: Reduce `--max_threads`

### Recovery from Interruptions

If the evaluation is interrupted:

1. **Check the log file** (`evaluation.log`) for the last processed item
2. **Re-run the same command** - the script will automatically skip already evaluated items
3. **Verify backups** are available in case of data corruption
