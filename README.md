# LLPut: Investigating Large Language Models for Bug Report-Based Input Generation

Welcome to the repository for our research paper, **"LLPut: Investigating Large Language Models for Bug Report-Based Input Generation."**  
This study explores how large language models (LLMs) perform in extracting precise, ordered command sequences from bug reports to facilitate bug reproduction.  

## Project Overview  

This repository contains all scripts, datasets, and supplementary materials used in our research. It is structured to ensure clarity, reproducibility, and ease of use for researchers and practitioners working on similar topics.  

## Repository Structure  

- **`/codes/`**  
  Contains scripts, input data, and output results used in the study:  
  - **`bleu-score/`**: BLEU score calculation module.  
    - **`bleu.ipynb`**: Jupyter Notebook for computing BLEU scores.  
    - **`input.csv`**: CSV file containing **206** bug report IDs, manually extracted inputs, and inputs recommended by LLaMA, Qwen, and Qwen-Coder.
  - **`ollama-run.py`**: Python script for running the models.

- **`/dataset/`**  
  Includes datasets generated and annotated during the study:
  - **`dataset.csv`**: Final dataset comprising **206** bug reports.  
  - **`descriptions.csv`**: Dataset containing **779** bug descriptions.  

- **`outputs/`**: Outputs from three models.

- **`data-collection-instructions.pdf`**
  Detailed guidelines for dataset collection and annotation.  
