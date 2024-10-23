# T5 Fine-Tuning for SQL Query Generation

This project demonstrates the fine-tuning of a pretrained T5 (Text-to-Text Transfer Transformer) model to translate natural language instructions into SQL queries. The goal is to develop a model that can understand user instructions for flight booking and generate the appropriate SQL query to retrieve the relevant data from a database.

## Overview

The project explores the sequence prediction task of converting natural language into SQL queries using the following approaches:
1. **Fine-tuning a pretrained T5 model**: Adapt the model to generate SQL queries from natural language instructions.
2. **Training a model from scratch**: Using the same architecture as T5, but starting with randomly initialized weights.
3. **Prompting with an LLM**: Utilize in-context learning and few-shot prompting with instruction-tuned large language models for zero- and few-shot SQL generation.

### Key Features:
- **Supervised fine-tuning** of the T5 model on a dataset of flight booking queries and their corresponding SQL queries.
- **Evaluation metrics**: The project measures the performance using Record F1, Record Exact Match, and SQL Query Exact Match, focusing on whether the generated SQL produces the correct database records.
- **Error analysis** to improve the fine-tuning process and minimize discrepancies between the generated and ground-truth SQL.

## Dataset

The dataset includes natural language instructions paired with ground-truth SQL queries for a flight booking database containing multiple tables (e.g., `airline`, `flight`, `restriction`). This data is split into training, development, and test sets.

## Results

- **Fine-tuned T5**: Achieved over 90% accuracy in generating SQL queries that return the correct database records.
- **Improved tokenization and data processing strategies**: Enhanced model performance by freezing parts of the T5 model during fine-tuning and optimizing the tokenization of SQL-specific keywords.

## Technologies Used:

- **Python**
- **HuggingFace Transformers (T5)**
- **PyTorch**
- **SQL**
- **Scikit-learn**

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/jihoonkim25/t5-sql-gen.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the training script:
    ```bash
    python train_t5.py
    ```

## Usage

To fine-tune the T5 model:
```bash
python train_t5.py --model_name t5-small --train_data data/train.nl --sql_data data/train.sql
```

For evaluation:
```bash
python evaluate.py --predicted_sql results/t5_ft_dev.sql --predicted_records records/t5_ft_dev.pkl
```
