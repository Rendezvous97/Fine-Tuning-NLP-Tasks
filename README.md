# Fine-Tuning NLP Tasks

A collection of fine-tuning experiments for various NLP tasks using different transformer models and datasets.

## Repository Structure

Each project follows the naming convention: `{task}-{base_model}-{dataset}.ipynb`

### Current Projects

| Task | Base Model | Dataset | Notebook | Status |
|------|------------|---------|----------|--------|
| Text Classification | DistilBERT | IMDB | [`text-classification-DISTILBERT-IMDB.ipynb`](text-classification-DISTILBERT-IMDB.ipynb) | ✅ Complete |

### Planned Projects

- `sentiment-analysis-ROBERTA-twitter.ipynb`
- `named-entity-recognition-BERT-conll2003.ipynb`
- `question-answering-BERT-squad.ipynb`
- `text-summarization-T5-cnn_dailymail.ipynb`
- `text-generation-GPT2-wikitext.ipynb`

## Current Project: Text Classification with DistilBERT on IMDB

### Overview
Fine-tuning DistilBERT for binary sentiment classification on the IMDB movie reviews dataset.

### Model Details
- **Base Model**: `distilbert-base-uncased`
- **Task**: Binary text classification (positive/negative sentiment)
- **Dataset**: IMDB movie reviews (25k train, 25k test)
- **Metrics**: Accuracy, F1-score, Precision, Recall

### Key Features
- Mixed precision training (fp16) for faster training
- Weights & Biases integration for experiment tracking
- Hugging Face Hub integration for model sharing
- Optimized for GPU training (tested on T4)

### Results
- **Training Time**: ~15-20 minutes on T4 GPU
- **Final Accuracy**: [To be updated after training]
- **Model Size**: ~67M parameters

## Setup Instructions

### Prerequisites
```bash
pip install transformers datasets evaluate accelerate wandb torch torchvision
```

### For Google Colab
```python
!pip install transformers datasets evaluate accelerate wandb -q
```

### Environment Setup
1. **Weights & Biases**: Create account at [wandb.ai](https://wandb.ai) and login
   ```python
   import wandb
   wandb.login()
   ```

2. **Hugging Face Hub**: Create account at [huggingface.co](https://huggingface.co) and login
   ```python
   from huggingface_hub import notebook_login
   notebook_login()
   ```

3. **Set tokenizer parallelism** (to avoid warnings):
   ```python
   import os
   os.environ["TOKENIZERS_PARALLELISM"] = "false"
   ```

## Hardware Requirements

### Minimum Requirements
- **RAM**: 8GB
- **Storage**: 2GB free space
- **GPU**: Optional but recommended

### Recommended Setup
- **GPU**: T4, V100, or better
- **RAM**: 16GB+
- **Storage**: 5GB+ free space

### Platform Options
- **Google Colab**: Free T4 GPU access
- **Kaggle Kernels**: Free GPU access
- **Local**: Apple Silicon Macs work well with MPS
- **Cloud**: AWS SageMaker, Lightning AI, etc.

## Training Configuration

### Optimized for GPU (T4)
```python
training_args = TrainingArguments(
    output_dir="tc-distilbert-imdb",
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_steps=50,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,  # Mixed precision for speed
    dataloader_num_workers=2,
    load_best_model_at_end=True,
    report_to="wandb",
)
```

### For CPU/Limited Memory
```python
training_args = TrainingArguments(
    output_dir="tc-distilbert-imdb",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    dataloader_num_workers=0,
    load_best_model_at_end=True,
)
```

## Experiment Tracking

All experiments are tracked using Weights & Biases:
- **Project**: `transformer-fine-tuning`
- **Metrics**: Training/validation loss, accuracy, F1-score
- **Hyperparameters**: Learning rate, batch size, epochs
- **System**: GPU utilization, memory usage

## Model Deployment

Trained models are automatically pushed to Hugging Face Hub:
- **Model Card**: Auto-generated with metrics and training details
- **Inference**: Can be used directly with `pipeline()` or `AutoModel`
- **Sharing**: Public models for community use

## Usage Example

```python
from transformers import pipeline

# Load the fine-tuned model
classifier = pipeline("text-classification", 
                     model="your-username/tc-distilbert-imdb")

# Make predictions
result = classifier("This movie was absolutely fantastic!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

## Contributing

Feel free to suggest new tasks, models, or datasets by opening an issue or pull request.

### Adding New Projects
1. Follow the naming convention: `{task}-{base_model}-{dataset}.ipynb`
2. Include comprehensive documentation in the notebook
3. Add experiment tracking with wandb
4. Update this README with project details
5. Ensure reproducibility with fixed random seeds

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Weights & Biases](https://wandb.ai/)
- [Google Colab](https://colab.research.google.com/) for free GPU access
- IMDB dataset creators and maintainers

---

**Note**: This repository is for educational and research purposes. Always check dataset licenses and model terms of use before commercial deployment. 