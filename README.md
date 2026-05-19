# Information Retrieval (IR) System

An implementation of a custom **Information Retrieval Engine** and **Document Clustering System** based on the Vector Space Model (VSM). This project was developed as part of the **Information Retrieval** laboratory course (Winter Semester 2025–2026) at the **Department of Computer Engineering and Informatics (CEID), University of Patras**.

The engine is evaluated on the classic **Cystic Fibrosis (C.F.) dataset**, which comprises a collection of **1,239 medical documents**, **20 test queries**, and a **ground-truth relevance list** (rated by medical specialists).

---

## Table of Contents
- [Project Architecture](#project-architecture)
- [File Tree & Descriptions](#file-tree--descriptions)
- [System Features](#system-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Evaluation Metrics](#evaluation-metrics)

---

## Project Architecture

The system is built upon five core experimental phases outlined in the project specification:
1. **Document Preprocessing & Indexing:** Building a custom Inverted Index mapping terms to documents and calculating Term Frequencies (TF) and Inverse Document Frequencies (IDF).
2. **Custom Vector Space Models (VSM):** Implementing two distinct tf-idf based term weighting systems (Classic TF-IDF vs. Cosine-Normalized `tfc-nfx` variant) and ranking using custom Cosine Similarity.
3. **Custom Evaluation Metrics:** Building a mathematical evaluation library from scratch (Precision, Recall, F1-score, Precision@k, and Precision-Recall Curves).
4. **Scikit-Learn TF-IDF Tuning:** Hyperparameter optimization via grid-search (>30 combinations) to compare custom VSM modules against a tuned `TfidfVectorizer`.
5. **Semantic Document Clustering:** Semantic grouping using **K-Means Clustering** applied to both TF-IDF sparse space and dense **SentenceTransformer** embeddings, optimized via **Silhouette Scores** and visualized using **PCA** dimensionality reduction.

---

## File Tree & Descriptions

```bash
informationRetrieval/
├── docs/                                  # Collection of 1,239 document text files (Cystic Fibrosis collection)
├── pyFiles/                               # Secondary/modular files split for Phase 2 functions
│   ├── analyshErwthsewn1.py               # Computes query tf-idf weights for VSM Method 1
│   ├── analyshErwthsewn2.py               # Computes query tf-idf weights for VSM Method 2
│   ├── analyshEurethriou1.py              # Generates document tf-idf vectors for VSM Method 1
│   ├── analyshEurethriou2.py              # Generates document tf-idf vectors for VSM Method 2
│   ├── findDocumentRanks1.py              # Computes Cosine Similarity and ranks docs for Method 1
│   ├── findDocumentRanks2.py              # Computes Cosine Similarity and ranks docs for Method 2
│   ├── printRelevancy1.py                 # Interactive terminal UI to print top matches for Method 1
│   └── printRelevancy2.py                 # Interactive terminal UI to print top matches for Method 2
├── textFiles/                             # Holds text datasets, index files, and experimental outputs
│   ├── Queries.txt                        # List of the 20 evaluation queries
│   ├── Relevant.txt                       # Ground-truth relevance mapping (Query ID -> Relevant Document IDs)
│   ├── inverted_index.json                # Inverted index file containing vocabulary details, TF, and IDF
│   ├── tfidfVectors.json / tfidfVectors2.json  # Precomputed VSM sparse vectors for Method 1 & 2
│   ├── queryVector.json / queryVector2.json    # Precomputed VSM query vectors for Method 1 & 2
│   ├── sortedRelevant1.json / sortedRelevant2.json # Retrieval rankings for each query
│   ├── bestModel.json                     # Hyperparameter details of the optimal Scikit-Learn VSM
│   └── results.txt                        # Detailed benchmark comparisons (Metrics & Processing times)
├── clustering.py                          # KMeans clustering pipeline comparing TF-IDF vs. Dense Sentence Embeddings
├── createEurethrio.py                     # Script to read documents, parse terms, and build the Inverted Index
├── evaluationMetricsFunctions.py         # Custom library implementing evaluation metrics and plotting PR curves
├── tfidf_tuning.py                        # Execution runner for hyperparameter grid search and multi-model benchmarking
├── vectorizer1.py                         # Single-file orchestrator for Custom VSM Method 1
├── vectorizer2.py                         # Single-file orchestrator for Custom VSM Method 2 (tfc-nfx weighting)
├── requirements.txt                       # List of required python libraries
└── README.md                              # Project documentation (this file)
```

### Key Python Script Roles:

*   **`createEurethrio.py` (Q1):** 
    Reads the document text corpus from `docs/`, sanitizes tokens, resolves potential indexing gaps caused by missing document IDs, calculates overall term frequencies and IDFs, and dumps the structure to `inverted_index.json`.
*   **`vectorizer1.py` & `vectorizer2.py` (Q2):** 
    Implement custom vector representations.
    *   **Method 1 (`vectorizer1.py`):** Uses standard tf-idf formulation.
    *   **Method 2 (`vectorizer2.py`):** Uses the normalized query formulation `(0.5 + 0.5 * (tf / max_tf)) * idf` combined with Cosine Normalization for document term weights (represented as the classic `tfc-nfx` scheme).
    Both rank matching documents per query using a custom-implemented Cosine Similarity.
*   **`evaluationMetricsFunctions.py` (Q3):** 
    Implements mathematical evaluation indicators without external model dependencies, including custom Precision, Recall, F1-score, Precision@k, and a dynamic plot routine for the Precision-Recall curve.
*   **`tfidf_tuning.py` (Q4):** 
    Performs grid search tuning over 48 permutations of Scikit-Learn's `TfidfVectorizer` parameters (`ngram_range`, `sublinear_tf`, `min_df`, `max_df`, `norm`) to find the best configuration based on **Mean Precision@10**. Saves comparative reports (custom models vs. scikit-learn best model including indexing/retrieval execution times) in `results.txt`.
*   **`clustering.py` (Q5):** 
    Applies the KMeans clustering algorithm to the document corpus under two feature spaces: sparse TF-IDF and dense embeddings extracted from Hugging Face's `sentence-transformers/all-MiniLM-L6-v2`. Computes optimal clusters ($k \in [2..10]$) using Silhouette Scores, reduces dimensionality to 2D using PCA, and plots the results.

---

## System Features
- **Zero-Dependency Core Metrics:** Core IR ranking, similarity calculations, and evaluation performance plots are written using base Python, Math, and NumPy to comply with academic strictness.
- **Advanced Term Weighting Schemes:** Comparative assessment of custom implementations (Classic vs. Cosine-Normalized `tfc-nfx`) and optimized library setups.
- **Deep Semantic Clustering:** Directly compares traditional lexicon-based sparse VSM clustering against modern, pre-trained transformer-based deep contextual embeddings.
- **Extensive Benchmarking:** High-fidelity timing instrumentation for both **indexing** and **retrieval** execution.

---

## Requirements
- Python 3.10+
- Dependencies listed in [requirements.txt](file:///c:/Users/dimit/Documents/Uni/7o/Information%20Retrieval/informationRetrieval/requirements.txt):
  - `scikit-learn`
  - `numpy`
  - `sentence-transformers`
  - `matplotlib`

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/jimfil/informationRetrieval/
   cd informationRetrieval
   ```
2. Install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### Step 1: Create the Inverted Index
Initialize the system vocabulary and document associations:
```bash
python createEurethrio.py
```

### Step 2: Run Custom Retrieval & Similarity Engine
You can run either custom VSM schemes. Run the files and press **Enter** to proceed, or **q** to quit:
*   **For Classic TF-IDF (Method 1):**
    ```bash
    python vectorizer1.py
    ```
*   **For Optimized tfc-nfx Weighting (Method 2):**
    ```bash
    python vectorizer2.py
    ```

### Step 3: Run Hyperparameter Tuning & Cross-Model Benchmarking
Execute the multi-model pipeline comparing custom implementations against scikit-learn variations:
```bash
python tfidf_tuning.py
```
*Outputs optimal grid parameters to `textFiles/bestModel.json` and a full evaluation scorecard to `textFiles/results.txt`.*

### Step 4: Perform Document Clustering
Run KMeans clustering to compare TF-IDF vs. transformer-based representation spaces:
```bash
python clustering.py
```
*This will open a detailed matplotlib window plotting PCA projections of the optimal clusters for both representations.*

---

## Evaluation Metrics
Each execution of `tfidf_tuning.py` computes:
*   **Precision@10:** Ratio of relevant items within the top 10 retrieved results.
*   **Recall:** Proportion of all truly relevant items that were successfully retrieved.
*   **F1-Score:** Harmonic mean of precision and recall.
*   **Precision-Recall Curve:** Visualizing the precision-recall trade-off across different thresholds.
