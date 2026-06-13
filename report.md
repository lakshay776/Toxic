# Toxic Comment Detection: A Comparative Study of Classical Sparse Models and Deep Sequence Models

## 1. Problem Statement

Automated toxicity detection is a critical challenge in modern natural language processing. At the scale of modern internet platforms, manual moderation is computationally and logistically impossible, necessitating robust machine learning systems capable of distinguishing between benign discourse and abusive behavior. However, this is a difficult modeling problem. Online communication is frequently informal, grammatically inconsistent, and adversarial. Malicious actors frequently employ obfuscated profanity, implicit toxicity, and context-dependent sarcasm to bypass automated filters. 

Furthermore, toxicity detection suffers from extreme class imbalance. In real-world datasets, the vast majority of user-generated content is benign. Consequently, a naive classifier that predicts all comments as non-toxic can easily achieve superficial accuracy approaching 90%. Building a useful system requires optimizing for metrics that accurately reflect the minority class, such as Precision, Recall, F1-Score, and Precision-Recall Area Under Curve (PR-AUC).

The primary objective of this project is to evaluate and compare two fundamentally different architectural paradigms for toxicity detection. Specifically, we investigate whether a classical, sparse lexical model (TF-IDF paired with a Multi-Layer Perceptron) can remain competitive with a more complex, sequence-based neural architecture (a Bidirectional LSTM). By evaluating these models under identical experimental conditions, we aim to understand the trade-offs between computational simplicity, lexical memorization, and dense sequence modeling.

## 2. Dataset and Preprocessing

The experiments in this study utilize the Jigsaw Toxic Comment Classification Challenge dataset. The dataset consists of 159,571 Wikipedia talk page edits originally labeled across six granular toxicity subtypes (toxic, severe_toxic, obscene, threat, insult, identity_hate). To focus on the core detection problem, the labels were collapsed into a binary classification task: comments containing any of the six flags were labeled as "toxic," while all others were labeled "non-toxic." 

The resulting dataset exhibits a severe class imbalance. Of the 159,571 training samples, 143,346 (~89.8%) are non-toxic, while only 16,225 (~10.2%) are toxic, representing an imbalance ratio of approximately 8.8:1. Because a trivial majority-class classifier would achieve nearly 89.8% accuracy, evaluation in this study relies heavily on PR-AUC, F1-scores, and threshold optimization.

Before feature extraction, the raw text underwent a standardized preprocessing pipeline. To reduce the vocabulary space and eliminate noise, all text was converted to lowercase. Uniform Resource Locators (URLs), internet protocol addresses, and non-alphanumeric special characters were stripped using regular expressions. While this normalization process successfully condenses the feature space, it introduces inherent limitations. Semantic information conveyed through capitalization or punctuation (such as exclamation marks indicating aggression) is lost. Additionally, aggressive normalization may alter adversarial spellings and remove useful orthographic signals, making it more difficult for downstream models to associate these modified patterns with toxicity.

## 3. Proposed Approach

To investigate the trade-offs between lexical and sequential modeling, we engineered two distinct pipelines.

### TF-IDF + Multi-Layer Perceptron (MLP)
The classical pipeline relies on Term Frequency-Inverse Document Frequency (TF-IDF) feature extraction. The vectorizer transforms the preprocessed text into high-dimensional, sparse lexical feature vectors, capturing the presence and frequency of unigrams and bigrams while discounting globally common words. These sparse vectors are then passed through a three-layer Multi-Layer Perceptron (MLP). The MLP learns non-linear relationships between the n-gram features to produce a final toxicity probability.

The primary advantage of the TF-IDF approach is computational efficiency and strong lexical memorization. It is highly effective when toxicity is explicitly linked to specific keywords or short phrases. However, it suffers from a fundamental limitation: order blindness. Although TF-IDF with unigram and bigram features captures local lexical relationships, it lacks explicit sequential modeling of long-range context and therefore struggles with negations and complex syntax where surrounding context alters the meaning of individual words., it struggles with negations and complex syntax where the context fundamentally alters the meaning of individual words.

### Bidirectional LSTM
The deep learning pipeline utilizes a Bidirectional Long Short-Term Memory (BiLSTM) network. In this architecture, tokenized sequences are passed through a randomly initialized, trainable embedding layer. The dense embeddings are then processed by the BiLSTM, which reads the sequence in both the forward and backward directions, allowing the network to capture contextual dependencies from the entire comment. The final hidden states are concatenated, passed through a dropout layer for regularization, and fed into a fully connected classification layer.

The BiLSTM was chosen specifically to address the limitations of the TF-IDF model. By capturing sequential information, it can theoretically differentiate between a word used maliciously and the same word used benignly based on surrounding context. To address the severe 8.8:1 class imbalance, the network was trained using a weighted Binary Cross-Entropy (BCE) loss function, where the positive (toxic) class was up-weighted by a factor of 8.83. This penalty encourages the optimization process to place greater importance on minority toxic samples, reducing the tendency to optimize majority-class accuracy at the expense of toxic recall.

## 4. Experimental Design

To ensure scientific rigor, both architectures were evaluated using an identical train-validation-test methodology. The models were trained on an 80% split, tuned on a 20% validation split, and ultimately evaluated on the official unseen Jigsaw test set of 63,978 comments.

### Baseline Ablations (Logistic Regression)
Before deploying complex architectures, we utilized a linear Logistic Regression model to establish interpretable empirical baselines on the validation set. 

First, an n-gram ablation study was conducted. While unigrams alone achieved the highest PR-AUC (0.8607), we selected unigrams and bigrams (`ngram_range=(1,2)`, PR-AUC: 0.8521) because bigrams capture critical two-word toxic phrases (e.g., 'kill you', 'shut up') that are linguistically meaningful. Extending to trigrams (`(1,3)`, PR-AUC: 0.8498) provided diminishing returns and inflated the vocabulary size unnecessarily.

Second, we conducted a class imbalance ablation. Training without class weighting heavily optimized for overall accuracy, resulting in an artificially high precision (0.9542) but a catastrophic minority-class recall (0.4470). Enabling `class_weight='balanced'` dramatically improved toxic recall to 0.7750 (with precision dropping to 0.6153) while maintaining a comparable PR-AUC (0.8029 vs 0.8107). This empirically validates the necessity of imbalance mitigation strategies for this dataset.

### Sequence Length Sensitivity
For the sequence-based BiLSTM, determining the appropriate context window is critical. We conducted an empirical sensitivity analysis to determine the optimal sequence length (`MAX_LEN`). Evaluating truncation lengths revealed that a `MAX_LEN` of 50 captured only ~61% of comment lengths, resulting in lower performance due to information loss. A length of 100 captured ~82%, showing marked improvement. Ultimately, a `MAX_LEN` of 230 was selected, providing coverage for ~95% of all comments in the dataset. Expanding the length to 250 increased computational cost and memory requirements while providing negligible performance gains.

### Threshold Optimization
A critical component of this study was the rejection of the default 0.5 classification boundary. Because of the imbalanced nature of the data, the default threshold rarely aligns with the optimal harmonic mean of precision and recall. Instead, both models underwent exhaustive threshold sweeping on the validation split. By extracting the precision-recall curve, we identified the exact operating points that maximized the F1-score while maintaining a minimum precision floor of 0.50. This optimization yielded final operational thresholds of approximately 0.7524 for the TF-IDF + MLP and 0.8134 for the BiLSTM.

### Error Analysis
To move beyond quantitative metrics, we conducted a qualitative error analysis on the official test set to understand the linguistic phenomena challenging both models. False positives frequently arose from "Identity Mention Bias," where benign comments discussing demographic or identity terms were incorrectly flagged. Additionally, both models struggled with non-toxic profanity (e.g., casual or quoting usage) and ambiguous language lacking conversational context. Conversely, false negatives were largely driven by obfuscated profanity that bypassed the vocabulary, indirect toxicity relying on sarcasm, and foreign language attacks that the English-centric models could not parse.

## 5. Results and Analysis

The final performance on the official Jigsaw Test Set is presented in Table 1.

**Table 1: Official Test Set Performance**

| Model        | Precision | Recall | F1   | ROC-AUC | PR-AUC |
| ------------ | --------- | ------ | ---- | ------- | ------ |
| TF-IDF + MLP | 0.5595    | 0.7847 | 0.6532| 0.9505  | 0.7389 |
| BiLSTM       | 0.4937    | 0.9092 | 0.6399| 0.9633  | 0.7749 |

Although the BiLSTM achieves a higher ROC-AUC and significantly improved recall (0.9092 vs 0.7847), the TF-IDF + MLP obtains a slightly higher F1-score (0.6532 vs 0.6399) due to its much stronger precision. This illustrates that ranking quality and a threshold-specific operating point capture different aspects of model performance.

The results reveal a nuanced trade-off rather than strict dominance by the deep learning architecture. Although the BiLSTM captures sequential context and achieves higher recall (0.9092 vs 0.7847) and higher ROC-AUC (0.9633 vs 0.9505), the TF-IDF + MLP remains highly competitive, achieving a comparable F1-score (0.6532 vs 0.6399).

A closer examination of the confusion matrices contextualizes these percentages. On the test set, the BiLSTM successfully detected 5,676 true positives but generated 5,820 false positives. In contrast, the TF-IDF + MLP detected 4,899 true positives while generating only 3,857 false positives. Practically, this means that while the BiLSTM detected 777 additional toxic comments, it did so at the cost of producing 1,963 additional false alarms.

This trade-off has significant implications for deployment. The preferred model depends entirely on application requirements. In a safety-critical moderation environment where protecting users is paramount, the higher recall of the BiLSTM is preferable. However, for a platform prioritizing community trust and minimizing the friction of false accusations against benign users, the higher precision and extreme computational efficiency of the TF-IDF + MLP may make it the superior choice.

## 6. Key Contributions

This project makes several empirical contributions to the study of automated moderation:
* **Architectural Comparison:** Implemented and compared classical sparse architectures against deep sequential NLP architectures under controlled experimental conditions.
* **Lexical Competitiveness:** Demonstrated that sparse lexical models remain highly competitive with deep networks on toxicity tasks, achieving comparable F1-scores despite significantly lower architectural and computational complexity.
* **Empirical Thresholding:** Conducted rigorous threshold optimization on validation data instead of relying on arbitrary 0.5 decision boundaries, adapting the models to imbalanced distributions.
* **Sequence Length Justification:** Performed sequence length sensitivity analysis to empirically justify the LSTM's sequence length configuration.
* **Qualitative Failure Analysis:** Analyzed real failure cases to classify model limitations, identifying identity bias and obfuscation as primary barriers to accurate classification.
* **Deployment-Oriented Evaluation:** Evaluated the models from a practical deployment perspective, weighing the costs of false positives against false negatives and computational latency.

Overall, the experiments demonstrate that increasing model complexity does not automatically translate into substantially better practical performance, and the appropriate moderation system depends on the trade-off between missed toxic content and false accusations.

## 7. Limitations and Future Work

While successful in comparing lexical and sequential approaches, this study has several limitations. The binary simplification of the Jigsaw dataset collapses nuanced abuse (e.g., differentiating between a generalized obscenity and a targeted identity threat) into a single label, losing category-specific information. Furthermore, the models evaluate comments in isolation, lacking the conversational context that often defines whether a statement is sarcastic, defensive, or genuinely toxic. Finally, the BiLSTM utilizes randomly initialized embeddings, which restricts its semantic understanding of rare words compared to models pre-trained on massive corpora.

Future work should address these limitations by extending the architectures to multi-label toxicity prediction. Replacing random embeddings with pre-trained contextual representations (e.g., FastText or GloVe) would likely improve robustness against typographical obfuscations. Furthermore, deploying Transformer-based architectures such as BERT or RoBERTa could vastly improve the modeling of long-range dependencies and implicit intent. Finally, exploring human-in-the-loop moderation systems could bridge the gap between algorithmic uncertainty and contextual human judgment.


## 8. Conclusion

This work compared classical sparse lexical representations against deep sequence models for automated toxicity detection. While the BiLSTM achieved higher recall and slightly improved ranking performance, the TF-IDF + MLP remained highly competitive with comparable F1-score and fewer false positives. The findings suggest that model selection for real-world moderation should be driven by the relative cost of false positives and false negatives rather than assuming that more complex neural architectures are universally superior.