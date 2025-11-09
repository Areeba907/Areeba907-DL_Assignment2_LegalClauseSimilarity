# 🧠 Deep Learning Assignment 2 – Legal Clause Similarity Detection  
## 📘 Project Overview
This project implements and compares two deep learning architectures — **Bidirectional LSTM (BiLSTM)** and **Attention Encoder** — for the task of **legal clause similarity detection**.  
The goal is to determine whether two clauses express the same legal meaning, an essential step in automating contract analysis and legal document comparison.

The dataset contains more than **150,000 clauses** extracted from **395 legal clause CSV files**, grouped by topics such as *acceleration*, *access to information*, *representations*, and *definitions*.  
Each pair of clauses is labelled as **similar (1)** or **different (0)**.

---

## ⚙️ Model Architectures

### 🔹 **1. BiLSTM**
- Embedding layer: 128-dimensional word embeddings  
- Bidirectional LSTM layer (hidden size = 128)  
- Clause representations combined using concatenation, absolute difference, and element-wise product  
- Fully connected layers with ReLU activation  
- Sigmoid output for binary classification  

### 🔹 **2. Attention Encoder**
- Multi-head self-attention layer (4 heads, embedding dim = 128)  
- Mean pooling to obtain contextual clause representation  
- Same combination and output structure as BiLSTM  

---

## 🧩 Dataset & Pre-processing
- 395 CSV files merged into a single dataset of **150,881 clauses**  
- Tokenisation with **NLTK Punkt**, lowercase normalisation, and padding to 128 tokens  
- Vocabulary size capped at **50,000 words**  
- Dynamic balanced pair generation (equal positive and negative examples)  
- Training/validation split: **60,000 / 15,000 pairs**

---

## 🧠 Training Configuration
| Hyperparameter | Value |
|----------------|--------|
| Epochs | 6 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | Binary Cross Entropy (BCELoss) |
| Framework | PyTorch |
| Environment | Google Colab (T4 GPU) |

---

## 📊 Results Summary

| Metric | **BiLSTM** | **Attention Encoder** |
|:--|:--:|:--:|
| Accuracy | **0.9995** | 0.9549 |
| Precision | **0.9996** | 0.9589 |
| Recall | **0.9995** | 0.9491 |
| F1-Score | **0.9995** | 0.9540 |
| ROC-AUC | **1.0000** | 0.9896 |

---

## 💬 Qualitative Examples

✅ **Correct Predictions**
- *Acceleration clauses*: Both models correctly recognised semantically identical clauses despite wording differences.  
- *Right-of-setoff clauses*: Correctly matched by both models.

❌ **Incorrect Predictions**
- Clauses with overlapping legal terms but different purposes (*e.g.*, “confidentiality” vs “publicity”) caused some misclassifications in the Attention Encoder.

---

## 📈 Training Progress
- **BiLSTM:** Rapid convergence with minimal loss by epoch 3  
- **Attention Encoder:** Gradual improvement; reached stable accuracy ≈ 95 % by epoch 6  

*(Attach your loss/accuracy plots here if desired.)*

---

## 🧩 Discussion
The BiLSTM achieved nearly perfect performance (99.95 % accuracy), proving highly effective for structured legal text due to its sequential modelling ability.  
The Attention Encoder delivered strong yet slightly lower accuracy (95.5 %), as mean-pooled self-attention can lose positional cues.  
Both models achieved **ROC-AUC > 0.98**, confirming excellent overall discrimination between similar and dissimilar clause pairs.

---

## 🔮 Future Work
- Incorporate **contextual embeddings** such as BERT or Legal-BERT  
- Explore **Siamese or contrastive training** for better generalisation  
- Test on larger, unbalanced or multilingual legal corpora  


