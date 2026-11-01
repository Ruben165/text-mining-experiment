# Text Mining Experiment

Personal project with three text mining cases on Indonesian text, all in `impl/Text_Mining_Experiment.ipynb`.

| Case | Data | Approach |
|---|---|---|
| Hate Speech Classification | `data_1A.csv` | Sentence-BERT embeddings (`firqaaa/indo-sentence-bert-base`) + MLP tuned with Optuna, per-label thresholds |
| Persona Analysis of Tweets | `data_2A.csv` | TF-IDF + LSA + K-Means (k chosen by silhouette score), LDA topics |
| Text Summarization | `data_3A.csv` | Extractive TextRank, sentence count chosen by ROUGE-L |

Each case ends with an inference example.

## Structure
```
data/            input CSVs
impl/            notebook
result/          plots and CSV outputs (created on run)
result/model/    best model of each case
result/cache/    cached sentence embeddings
```

## Setup
```
pip install numpy pandas matplotlib nltk scikit-learn keras tensorflow optuna sentence-transformers rouge wordcloud networkx joblib
```
Run the notebook from `impl/`, starting with the Shared Setup cell. Runs on CPU; Case 1 is the slowest (first run downloads the embedding model).
