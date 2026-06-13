# Toxic Comment Detection: Classical vs Deep Learning Approaches

An end-to-end NLP system for detecting toxic online comments using the **Jigsaw Toxic Comment Classification dataset**, comparing a classical sparse-feature approach (**TF-IDF + Multi-Layer Perceptron**) against a sequence-based deep learning approach (**BiLSTM**).

The project goes beyond raw accuracy and investigates a fundamental NLP question:

> Does increasing model complexity actually improve toxic comment detection, or do simpler lexical models remain competitive?

The study explores model trade-offs through precision-recall analysis, threshold optimization, sequence length sensitivity experiments, and qualitative error analysis.

---

## Key Highlights

* Built two independent toxicity detection pipelines:

  * TF-IDF + Multi-Layer Perceptron
  * Bidirectional LSTM with trainable embeddings

* Conducted a complete experimental analysis including:

  * Class imbalance analysis (≈ 8.8:1 non-toxic to toxic ratio)
  * Precision-Recall and ROC analysis
  * Threshold optimization instead of using the default 0.5 decision boundary
  * Sequence length sensitivity study (`MAX_LEN = 50 → 250`)
  * Misclassification analysis of false positives and false negatives
  * Deployment-oriented comparison of latency, model size, and error costs

* Fully reproducible Google Colab workflow:

  * One-click `Runtime → Run All`
  * Automatic dependency installation
  * Dataset verification
  * Git LFS dataset retrieval fallback

---

# Dataset

Dataset: **Jigsaw Toxic Comment Classification**

Training samples: 159,571 comments

Binary toxicity distribution:

| Class     |   Count | Percentage |
| --------- | ------: | ---------: |
| Non-toxic | 143,346 |      89.8% |
| Toxic     |  16,225 |      10.2% |

The severe class imbalance makes accuracy misleading. A naive model predicting every comment as non-toxic would already achieve approximately **89.8% accuracy**, motivating the use of Precision, Recall, F1 score, ROC-AUC, and PR-based analysis.

---

# Model Architectures

## TF-IDF + MLP

Pipeline:

```
Raw Comment
      |
      v
TF-IDF Vectorizer
      |
      v
Sparse Feature Vector
      |
      v
Multi-Layer Perceptron
      |
      v
Toxic Probability
```

Advantages:

* Extremely fast inference (~0.011 ms/sample)
* Strong lexical understanding
* Simple and interpretable

Limitations:

* No explicit understanding of long-range word order

---

## BiLSTM

Pipeline:

```
Raw Comment
      |
      v
Tokenization
      |
      v
Embedding Layer
      |
      v
Bidirectional LSTM
      |
      v
Dense Classification Layer
      |
      v
Toxic Probability
```

Design choices:

* Trainable embeddings
* Bidirectional context modeling
* Weighted BCE loss to address the ~8.8:1 class imbalance
* Sequence length selected through empirical sensitivity analysis

Final sequence length:

```
MAX_LEN = 230
```

This captures approximately 95% of comments while avoiding unnecessary computational overhead.

---

# Experimental Results

## Official Test Set Performance

| Model        | Precision | Recall |   F1 | ROC-AUC |
| ------------ | --------: | -----: | ---: | ------: |
| TF-IDF + MLP |    0.5595 | 0.7847 |0.6532|  0.9505 |
| BiLSTM       |    0.4937 | 0.9092 |0.6399|  0.9633 |

A key observation is that the BiLSTM does **not overwhelmingly outperform** the classical model.

Instead, the results reveal a meaningful trade-off:

* BiLSTM detects 777 additional toxic comments.
* It also produces 1,963 additional false positives.
* TF-IDF achieves slightly higher precision with significantly simpler modeling assumptions.

This highlights an important practical NLP insight:

> For toxicity detection, lexical signals are extremely strong, allowing sparse TF-IDF representations to remain highly competitive with more complex deep sequence models.

---

# Threshold Optimization

The default classification threshold of 0.5 was not assumed.

Each model underwent threshold sweeping to optimize the Precision-Recall trade-off.

Chosen thresholds:

| Model        | Threshold |
| ------------ | --------: |
| TF-IDF + MLP |    0.7524 |
| BiLSTM       |    0.8134 |

This allows the models to operate at application-specific trade-offs between:

* Missing harmful content (false negatives)
* Incorrectly flagging safe content (false positives)

---

# Error Analysis

Representative failure modes include:

## False Positives

* Identity-related terms appearing in non-toxic contexts.
* Profanity used casually or positively.
* Ambiguous language lacking conversational context.

Example:

```
"he's gay too"
```

A possible explanation is that the model associates certain identity terms with toxicity due to correlations present in the training data.

---

## False Negatives

Examples include:

* Obfuscated profanity:

  * `fckin stupid hoee`

* Foreign language abuse.

* Implicit insults without explicit toxic keywords.

These cases highlight limitations of both lexical and sequence-based approaches.

---

# Repository Structure

```text
Toxic_Comment_Detection/
│
├── toxic_detection_final.ipynb
│   └── Complete reproducible research notebook
│
├── report.pdf
│   └── Final project report
│
├── models/
│   ├── toxic_lstm.pt
│   ├── toxic_tfidf_mlp.pkl
│   ├── tfidf_vectorizer.pkl
│   └── word2idx.json
│
├── api/
│   └── FastAPI inference service
│
├── requirements.txt
│
└── data/
    └── Dataset documentation
```

---

# Reproducibility

The notebook is designed to execute from a fresh Google Colab environment.

Steps:

1. Upload `toxic_detection_final.ipynb` to Google Colab.
2. Select `Runtime → Run All`.
3. The notebook automatically:

   * Clones the repository.
   * Installs dependencies.
   * Retrieves datasets using Git LFS if required.
   * Executes all experiments.
   * Generates figures, metrics, and final artifacts.

No manual preprocessing or configuration is required.

---

# Deployment

A FastAPI inference layer is included for serving trained models.

The API supports:

* Loading serialized model artifacts.
* Preprocessing incoming comments.
* Returning toxicity probabilities and predictions.

---

# Limitations

Current limitations include:

* Binary toxicity classification loses fine-grained labels.
* No conversational context between comments.
* Difficulty handling sarcasm and implicit toxicity.
* Vulnerability to adversarial misspellings and unseen linguistic patterns.

---

# Future Improvements

Potential extensions:

* Multi-label toxicity prediction.
* Transformer-based models such as BERT.
* Pre-trained word embeddings.
* Human-in-the-loop moderation systems.
* More robust handling of adversarial text.

---

# Course Context

This project was developed as part of an Advanced Machine Learning / Deep Learning coursework project focused on understanding not only **how to build toxicity classifiers**, but also **why different architectures succeed or fail under realistic constraints**.
