# M25CSA029-A2

##  Project Structure
* `nlu_assgn_q1.py`: PDF text extraction, cleaning, and Skip-gram Word2Vec implementation.
* `nlu_assgn_q2.py`: Vanilla RNN, BiLSTM, and Attention-based name generation.
* `TrainingNames.txt`: Dataset of 1,000 Indian names (Required for Problem 2).
* `document1.pdf`, `document2.pdf`, `document3.pdf`....: Source documents (Required for Problem 1).

---

##  Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed. You will need the following libraries:

```bash
pip install torch pandas numpy pymupdf wordcloud matplotlib scikit-learn
```

### 2. Setting Up Data
* **For Problem 1:** Place your three source PDF files in the same directory as the script.
* **For Problem 2:** Ensure `TrainingNames.txt` is present in the root directory.

---

##  Running the Code

### Problem 1: Word2Vec Semantic Analysis
This script extracts text from the PDFs, trains a Skip-gram model from scratch, and outputs semantic similarities and a t-SNE visualization.

```bash
python problem1_word2vec.py
```
**Outputs:**
* `cleaned_corpus.txt`: The processed text used for training.
* t-SNE Cluster Plot (GUI window).
* Top-10 frequent words and a 300-dimension vector for a sample word.

### Problem 2: Character-Level Name Generation
This script trains three models (RNN, BiLSTM, Attention) and compares their performance using Novelty and Diversity metrics.

```bash
python problem2_name_gen.py
```
**Outputs:**
* `Evaluation_Metrics.txt`: Quantitative results (Params, Novelty, Diversity).
* `Generated_Samples.txt`: 100 generated names per model architecture.

---

##  Model Summaries

### Problem 1: Word2Vec
* **Architecture:** Skip-gram implemented with `nn.Embedding` and a Linear output layer.
* **Objective:** Predicts context words given a target word using Cross-Entropy loss.

### Problem 2: Name Generation
* **Vanilla RNN:** Standard recurrence with $Tanh$ activation (~26k params).
* **BiLSTM:** Scratch-built with manual Input, Forget, and Output gates (~91k params).
* **Attention RNN:** Enhances standard RNN by calculating a context vector across all previous hidden states (~43k params).

---

##  Key Findings
* **Best Performer:** The **BiLSTM** produced the most phonetically realistic Indian names while maintaining high novelty.
* **Efficiency:** The **Vanilla RNN** is the smallest model (~0.1 MB), making it the most lightweight but prone to "forgetting" long-distance character dependencies.
