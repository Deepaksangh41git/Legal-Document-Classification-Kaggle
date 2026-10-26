# Legal Document Classification using Legal-BERT

A Natural Language Processing (NLP) project developed for the **IIT Madras Data Science Learning Platform (DLP) NPPE-1 T2 2026 competition**. The project uses a domain-specific Transformer model to classify legal documents through text preprocessing, fine-tuning, and chunk-based inference.

## Overview

Legal documents often contain lengthy, complex text and domain-specific terminology. Traditional machine learning approaches may struggle to capture the contextual meaning of legal language.

This project explores **Legal-BERT**, a BERT-based language model designed for legal-domain text, to build a document classification pipeline using PyTorch and Hugging Face Transformers.

The pipeline covers data preprocessing, model training, validation, inference, and submission file generation.

## Key Features

- **Legal-domain NLP:** Uses Legal-BERT for contextual understanding of legal documents.
- **Text preprocessing:** Cleans European legal documents by removing unnecessary formatting, boilerplate sections, and redundant whitespace.
- **Long-document processing:** Splits lengthy documents into overlapping token chunks to handle text exceeding the model's maximum input length.
- **Transformer fine-tuning:** Trains a sequence classification model using PyTorch and Hugging Face Transformers.
- **Validation:** Uses a stratified train-validation split to preserve class proportions.
- **Reproducibility:** Sets random seeds for consistent experimental runs.
- **Efficient preprocessing:** Caches cleaned training and test data to avoid repeated preprocessing.
- **Automated inference:** Generates class predictions for unseen documents.
- **Submission generation:** Produces a `submission.csv` file aligned with the competition's sample submission format.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Model training and inference |
| Hugging Face Transformers | Pretrained Transformer models and tokenization |
| Legal-BERT | Legal document classification |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Data splitting and evaluation |
| tqdm | Progress monitoring |
| Kaggle Notebooks | Competition environment and GPU execution |

## Model Architecture

**Base model:** `nlpaueb/legal-bert-base-uncased`

Legal-BERT is a BERT-based language model pretrained on legal-domain text. Its domain-specific representations make it a suitable candidate for legal document classification.

The model is fine-tuned for sequence classification using the following configuration:

| Parameter | Value |
|---|---|
| Base model | Legal-BERT Base Uncased |
| Maximum sequence length | 512 tokens |
| Batch size | 16 |
| Learning rate | 3e-5 |
| Training epochs | 1 |
| Weight decay | 0.01 |
| Warmup ratio | 0.06 |
| Gradient accumulation steps | 16 |
| Gradient clipping | 1.0 |
| Label smoothing | 0.05 |
| Random seed | 42 |
| Validation split | 20% |

These are the configuration values in the current notebook and may be changed during experimentation.

## Project Workflow

### 1. Data Loading

Load the competition's training, test, and sample submission CSV files.

The training dataset contains the legal document text and its corresponding target label. The test dataset contains documents requiring predictions.

### 2. Text Preprocessing

Apply custom cleaning rules to the legal documents, including:

- Removing unnecessary formatting markers.
- Cleaning selected European legal publication references.
- Removing specified boilerplate sections.
- Normalizing whitespace.
- Converting document text into a consistent format.

The cleaned data is cached locally to accelerate subsequent runs.

### 3. Train-Validation Split

Split the labeled dataset into training and validation subsets using stratified sampling.

This helps preserve the distribution of target classes across both subsets and provides a basis for evaluating the model during training.

### 4. Tokenization

Use the Legal-BERT tokenizer to convert document text into token IDs and attention masks.

The maximum sequence length is set to 512 tokens, matching the model's input configuration.

### 5. Model Fine-Tuning

Fine-tune the pretrained Legal-BERT sequence classification model using PyTorch.

The training pipeline incorporates:

- Gradient accumulation.
- Learning rate scheduling with warmup.
- Weight decay.
- Gradient clipping.
- Label smoothing.
- Model checkpointing and training-state tracking.

The current notebook configuration uses one training epoch to accommodate time constraints.

### 6. Chunk-Based Inference

Long documents may exceed the model's input-length limit. To address this, the inference pipeline:

1. Tokenizes each document.
2. Divides the token sequence into overlapping chunks.
3. Processes the chunks using the fine-tuned model.
4. Aggregates chunk-level predictions into document-level probabilities.
5. Selects the predicted class from the resulting probabilities.

The implementation limits the number of chunks processed per document to manage computational requirements.

### 7. Submission Generation

Generate predictions for the competition's test dataset and save them to `submission.csv`.

The output columns and document ID ordering are aligned with the provided sample submission file.

## Repository Structure

The repository can be organized as follows:

```text
legal-document-classification/
│
├── README.md
├── 23f2004126-dlp-26t2-nppe1-ipynb.ipynb
├── requirements.txt
└── .gitignore
```

The notebook contains the preprocessing, model training, inference, and submission-generation pipeline.

*Note: The structure above is a suggested repository layout. Include only the files that actually exist in your repository.*

## Getting Started

### Prerequisites

- Python 3.10 or later
- Access to a compatible GPU for practical model fine-tuning
- A Kaggle account with access to the competition dataset

### Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd legal-document-classification
```

Install the required libraries:

```bash
pip install numpy pandas scikit-learn torch transformers tqdm
```

Additional packages or version constraints may be required depending on the notebook environment.

### Dataset Setup

The notebook expects the competition files at the following Kaggle paths:

```text
/kaggle/input/competitions/dlp-nppe-1-t-22026/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Attach the competition dataset to your Kaggle notebook or update the file paths to match your local environment.

**Important:** The competition dataset is not included in this repository. Access may be subject to the competition's rules and dataset licensing terms.

### Running the Project

1. Open the notebook in Kaggle or a compatible Jupyter environment.
2. Configure the dataset paths and GPU runtime.
3. Execute the notebook cells in order.
4. Allow preprocessing and model fine-tuning to complete.
5. Run the inference cells to generate predictions.
6. Retrieve the generated `submission.csv` file.

The current training configuration uses a GPU device. A compatible CUDA-enabled environment is therefore required unless the notebook is modified to support CPU execution.

## Evaluation

The notebook uses a stratified validation split to support model evaluation during training.

The final competition performance should be reported using the metric specified by the competition organizers.

| Metric | Result |
|---|---|
| Validation accuracy | Not reported in the current README |
| Competition leaderboard score | Not reported in the current README |

Actual evaluation results should be added after running the notebook and checking the validation output or competition leaderboard.

## Challenges and Considerations

### Long Legal Documents

Legal documents can exceed Transformer input-length limits. Overlapping chunks help preserve local context, but the chunk limit can prevent the model from processing every part of a very long document.

### Domain-Specific Language

Legal terminology and document structure can be difficult to interpret using general-purpose language models. Legal-BERT provides a domain-adapted starting point for this task.

### Computational Constraints

Transformer fine-tuning can be computationally expensive. Gradient accumulation, cached preprocessing, and controlled training configurations help manage resource requirements.

### Generalization

A stratified validation split provides an initial estimate of model performance, but leaderboard performance may differ. Further experiments can investigate class imbalance, training duration, chunk aggregation, and model hyperparameters.

## Future Improvements

Potential directions for further experimentation include:

- Training for additional epochs with appropriate validation and early stopping.
- Comparing Legal-BERT against general-purpose BERT-based models.
- Experimenting with different learning rates and batch sizes.
- Evaluating alternative strategies for aggregating predictions across document chunks.
- Investigating class imbalance and per-class performance.
- Using cross-validation to obtain more robust evaluation estimates.
- Comparing validation metrics with public leaderboard results.
- Optimizing inference speed and GPU memory usage.

These are proposed improvements rather than completed experiments.

## Disclaimer

This project is intended for educational and machine learning experimentation as part of a data science competition.

The model's predictions should not be treated as legal advice or as a substitute for professional legal review.

## Author

**Deepak Singh**

B.Tech in Computer Science | BS in Data Science and Applications, IIT Madras

**Areas of interest:** Machine Learning, Natural Language Processing, Deep Learning, and Applied AI.
