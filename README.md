# Spam/Toxic Comment Detection

This repository contains the implementation of a text classification system designed to detect toxic comments, built using the Jigsaw Toxic Comment Classification Challenge dataset. 

The core of this project satisfies the following objectives:
1. **Building classifiers** to detect toxic comments.
2. **Comparing** a Multi-Layer Perceptron (MLP) trained on TF-IDF features against a Bidirectional LSTM neural network.
3. **Analyzing** precision-recall trade-offs.
4. **Examining** misclassified examples to understand model failures.

---

## 📊 Model Comparison & Results

Both models were trained using a two-phase approach: first tuning the classification threshold on a Validation Split (80/20) to optimize the F1-Score, and then retraining on the full dataset before evaluating on the Official Test Set (63,978 samples).

### 1. Bidirectional LSTM (PyTorch)
The LSTM processes sequences of words via learned embeddings, giving it a strong grasp of context and word order.

* **ROC-AUC (Official Test):** 0.9550
* **Precision (Toxic):** 0.53
* **Recall (Toxic):** 0.83
* **F1-Score (Toxic):** 0.65
* **Chosen Threshold:** 0.8460

### 2. TF-IDF + MLPClassifier (Scikit-Learn)
The TF-IDF model evaluates text based on the frequency of up to 50,000 unigrams and bigrams, passed through a 3-layer dense neural network.

* **ROC-AUC (Official Test):** 0.9505
* **Precision (Toxic):** 0.56
* **Recall (Toxic):** 0.78
* **F1-Score (Toxic):** 0.65
* **Chosen Threshold:** 0.7524

### 📈 Precision-Recall Trade-off Analysis
Because the dataset is heavily imbalanced (most comments are non-toxic), standard accuracy is misleading. We utilized `precision_recall_curve` to find the exact classification thresholds (0.8460 for LSTM, 0.7524 for TF-IDF) that maximized the F1-score. 

By slightly lowering our thresholds, we intentionally accepted a drop in **Precision** (leading to more False Positives) in exchange for a significant boost in **Recall** (catching the vast majority of genuinely toxic comments). For moderation systems, missing a highly toxic comment (False Negative) is generally considered worse than accidentally flagging a benign comment for human review (False Positive).

### 🔍 Error Analysis: Understanding Failures
We manually analyzed the false positives and false negatives from the official test set:

* **TF-IDF Vulnerabilities (False Positives):** The MLP heavily over-indexes on specific "toxic" vocabulary. If a user writes a benign, academic, or quoting comment that contains words like "idiot" or "kill" (e.g., discussing a Wikipedia article about a crime), the TF-IDF model often falsely flags it as toxic because it lacks contextual awareness.
* **LSTM Vulnerabilities (False Negatives):** The LSTM struggles with out-of-vocabulary (OOV) tokens. If a user obfuscates profanity (e.g., `f*ck` or `sh1t`), the tokenizer maps it to the `<UNK>` (Unknown) token. Without the explicit word embedding, the LSTM often fails to recognize the threat and falsely clears the comment as non-toxic.

---

## 🚀 FastAPI Inference API

The project includes a production-ready API to serve real-time predictions using the trained PyTorch LSTM model.

### Running the Server
1. Ensure your trained PyTorch model (`toxic_lstm (1).pt`) and vocabulary mapping (`word2idx.json`) are in the root directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   python -m uvicorn main:app --reload --port 8005
   ```

### Endpoints

#### `POST /predict`
Analyzes a string of text and returns a toxicity prediction.

**Request Body (JSON):**
```json
{
  "text": "Your comment goes here."
}
```

**Response (JSON):**
```json
{
  "text": "Your comment goes here.",
  "is_toxic": true,
  "confidence": 0.9854
}
```

**How to Test:**
You can test the API instantly via the interactive Swagger UI by navigating to `http://localhost:8005/docs` while the server is running.
